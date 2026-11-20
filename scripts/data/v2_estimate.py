"""Cost and wall-time model for the v2 world generator
(`01-spec/world-generator-v2-2026-09-22.md` section 8, section 9's WP11
row), built from MEASURED call logs rather than the spec's pre-build
guesses.

What it reads (all under `--var-root`, default the repo's `var/`; read only):

| Measure | Log | Why these runs |
|---|---|---|
| Writer cost, tokens and seconds per document, by kind | `data/synth_v2/smoke_wp4fix/{.,merged,merged3_0050427,voice_f425a12,nofacts_c064ff6,length_b089d43}/calls.jsonl` | every WP4 smoke on the merged code (WP4 review, section 5) |
| WP4 labeled share of planted facts | the same runs' `run_report.json` | |
| Teacher (Kimi K2) cost and seconds per document, by kind; WP5 regeneration share; candidates per document | `data/synth_v2/smoke_wp4fix/**/wp5_measure/` | WP5's blind pass on those documents, no regeneration, no audit |
| Audit teacher per document; share kept after a WP5 regeneration | `data/synth_v2/smoke_wp5/*/` | the WP5 smoke (built on WP4 at 6606c6f, an older prompt) |
| WP5 regeneration share under the current rule (only a world-breaking candidate sends a document back; 2026-09-22 cost decision) | every WP5 run above plus `data/synth_v2/smoke_pilot_prep/*/*/completeness/` whose `qa_report.json` pass version is `CURRENT_POLICY_PASS` or later | runs of the rule the runner uses |
| Label judge per document | `label_judge.jsonl` in those WP5 runs | the judge runs on every shipped document's silver labels |
| Near-duplicate writer calls | `data/synth_v2/smoke_wp6/calls.jsonl` | the WP6 smoke |
| Three-judge dispute cost and seconds | `bench/wp9-smoke-2026-09-22/recheck-ticket/disputes.jsonl` | WP9's run after its `max_tokens` fix |
| Local MLX decode speed | `bench/local-teachers-2026-09-22/*.log` (`mlx call` lines) | the only local timing on this machine |

Seconds per call are the gap to the next call's start in the same log (the
runner is sequential, so this is wall time including verification). A gap
over `PAUSE_S` means the log holds two runs; that call's seconds are the
median of its kind instead. Document counts per kind come from simulating
and planning every world of the round (`corpus.generator.v2.rounds`), which
is deterministic and makes no call.

Not measured, and said so in the output: the questions stage of a test
world (WP10 is not built; the spec's figure is carried), the local prefill
speed (an assumption, `--local-prefill-tps`), and whether local writers
need more repairs than paid ones (assumed the same).

The runner's order is WP4, WP5, WP6: the teacher and the audit read every
WP4 document (`completeness.AUDIT_SHARE`), never a near-duplicate, and the
per-type cap applies at plan time, so the planted facts counted here are the
capped ones.

    uv run python scripts/data/v2_estimate.py [--var-root DIR] [--worlds round1] [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from corpus.generator.v2.completeness import AUDIT_SHARE  # noqa: E402
from corpus.generator.v2.planner import DocRole, plan_world  # noqa: E402
from corpus.generator.v2.rounds import PACKS_DIR, WorldSpec, parse_worlds  # noqa: E402

__all__ = [
    "CallRow",
    "Measured",
    "PerDoc",
    "WorldCounts",
    "estimate_world",
    "load_calls",
    "measure",
    "world_counts",
]

# The longest single call in the smoke logs took about 270 s (a writer that
# looped); the WP6 log holds two runs 327 s apart.
PAUSE_S = 300.0
# WP9's disputes.jsonl stamps every dispute with one time, so judge time is
# not measurable from it; WP9's build note says several hundred disputes
# take about an hour, about 12 s each.
UNMEASURED_JUDGE_SECONDS_PER_DISPUTE = 12.0

# Spec section 8, the modeled numbers this replaces (for the comparison line).
SPEC_ROUND1_USD = 36.56
SPEC_ROUND1_USD_MEASURED_TEACHER = 48.92
SPEC_TRAIN_WORLD_USD = 1.60
SPEC_TEST_WORLD_USD = 4.70
# Spec section 8's test-world questions line; WP10 is not built, so there is
# nothing to measure yet.
SPEC_QUESTIONS_USD_PER_TEST_WORLD = 0.86
# The first WP5 pass version that labels consistent candidates instead of
# regenerating (spec Build notes, WP5 after the WP4 merge, 2026-09-22).
CURRENT_POLICY_PASS = "wp5-2026-09-22.10"
# Assumed, not measured: MLX prompt processing on this Mac (the one local
# timing has 94% of its prompt cached, so it cannot separate prefill).
DEFAULT_LOCAL_PREFILL_TPS = 250.0

WRITER_RUNS = (
    "data/synth_v2/smoke_wp4fix",
    "data/synth_v2/smoke_wp4fix/merged",
    "data/synth_v2/smoke_wp4fix/merged3_0050427",
    "data/synth_v2/smoke_wp4fix/voice_f425a12",
    "data/synth_v2/smoke_wp4fix/nofacts_c064ff6",
    "data/synth_v2/smoke_wp4fix/length_b089d43",
)
TEACHER_MEASURE_GLOB = "data/synth_v2/smoke_wp4fix/**/wp5_measure"
WP5_SMOKE_GLOB = "data/synth_v2/smoke_wp5/*"
WP5_PILOT_PREP_GLOB = "data/synth_v2/smoke_pilot_prep/*/*/completeness"
NEARDUP_LOG = "data/synth_v2/smoke_wp6/calls.jsonl"
JUDGE_LOG = "bench/wp9-smoke-2026-09-22/recheck-ticket/disputes.jsonl"
MLX_LOG_GLOB = "bench/local-teachers-2026-09-22/*.log"

_KIND_RE = re.compile(r"^S-(?P<kind>[a-z_\-]+?)-\d+$")
_MLX_RE = re.compile(r"mlx call \d+: in=(\d+) cached=(\d+) out=(\d+) ([\d.]+)s")
KEPT_STATUSES = frozenset({"accepted", "clean_negative", "relabeled_positive"})


# --------------------------------------------------------------------------
# Reading logs
# --------------------------------------------------------------------------


@dataclass
class CallRow:
    log: str
    plan_id: str
    kind: str | None
    call_kind: str  # initial / repair / regenerate for writers; completeness / audit for teachers
    model: str
    cost_usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempt: int = 0
    error: bool = False
    at: datetime | None = None
    seconds: float | None = None
    run: int = 0  # index of the run inside its log (a log can hold appended runs)


def kind_of(plan_id: str) -> str | None:
    """`S-call-00002` -> `call`; `S-hard-negative-00004` -> `hard-negative`."""
    m = _KIND_RE.match(plan_id)
    return m.group("kind") if m else None


def _parse_at(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def load_calls(path: Path, *, label: str | None = None) -> list[CallRow]:
    """Every row of one call log, with `seconds` from the gap to the next
    call's start and `run` split at gaps over `PAUSE_S`."""
    rows: list[CallRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(
            CallRow(
                log=label or str(path),
                plan_id=r["plan_id"],
                kind=kind_of(r["plan_id"]),
                call_kind=r.get("call_kind") or r.get("role") or "unknown",
                model=r.get("model", ""),
                cost_usd=float(r.get("cost_usd") or 0.0),
                prompt_tokens=int(r.get("prompt_tokens") or 0),
                completion_tokens=int(r.get("completion_tokens") or 0),
                attempt=int(r.get("attempt") or 0),
                error=bool(r.get("error")),
                at=_parse_at(r.get("at")),
            )
        )
    rows.sort(key=lambda c: (c.at is None, c.at or datetime.min))
    run = 0
    for i, row in enumerate(rows):
        row.run = run
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if row.at is None or nxt is None or nxt.at is None:
            continue
        gap = (nxt.at - row.at).total_seconds()
        if gap > PAUSE_S:
            run += 1
        elif gap >= 0:
            row.seconds = gap
    medians: dict[str, float] = {}
    for kind in {r.call_kind for r in rows}:
        known = [r.seconds for r in rows if r.call_kind == kind and r.seconds is not None]
        if known:
            medians[kind] = statistics.median(known)
    every = [r.seconds for r in rows if r.seconds is not None]
    fallback = statistics.median(every) if every else 0.0
    for row in rows:
        if row.seconds is None:
            row.seconds = medians.get(row.call_kind, fallback)
    return rows


@dataclass
class PerDoc:
    """Means per document over `n` documents (a document in one run)."""

    n: int = 0
    cost_usd: float = 0.0
    calls: float = 0.0
    prompt_tokens: float = 0.0
    completion_tokens: float = 0.0
    seconds: float = 0.0


def per_doc(rows: Iterable[CallRow]) -> PerDoc:
    groups: dict[tuple[str, int, str], list[CallRow]] = defaultdict(list)
    for r in rows:
        groups[(r.log, r.run, r.plan_id)].append(r)
    if not groups:
        return PerDoc()
    n = len(groups)
    return PerDoc(
        n=n,
        cost_usd=sum(r.cost_usd for g in groups.values() for r in g) / n,
        calls=sum(len(g) for g in groups.values()) / n,
        prompt_tokens=sum(r.prompt_tokens for g in groups.values() for r in g) / n,
        completion_tokens=sum(r.completion_tokens for g in groups.values() for r in g) / n,
        seconds=sum(r.seconds or 0.0 for g in groups.values() for r in g) / n,
    )


def per_doc_by_kind(rows: Sequence[CallRow]) -> dict[str, PerDoc]:
    kinds = sorted({r.kind for r in rows if r.kind})
    return {k: per_doc(r for r in rows if r.kind == k) for k in kinds}


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------


@dataclass
class Rate:
    value: float
    n: int  # how many units it was measured on

    def __float__(self) -> float:
        return self.value


@dataclass
class Measured:
    writer_by_kind: dict[str, PerDoc]
    writer_all: PerDoc
    teacher_by_kind: dict[str, PerDoc]
    teacher_all: PerDoc
    audit: PerDoc
    neardup: PerDoc
    judge_per_dispute: PerDoc
    wp4_labeled_share: Rate
    wp5_regeneration_share: Rate
    wp5_kept_after_regeneration: Rate
    wp5_candidates_per_doc: Rate
    local_decode_tps: Rate
    # The rule the runner uses (`CURRENT_POLICY_PASS`); `wp5_regeneration_share`
    # is the first-pass rate under the earlier section 3.4 rule, kept for comparison.
    wp5_regeneration_share_current: Rate = field(default_factory=lambda: Rate(0.0, 0))
    label_judge: PerDoc = field(default_factory=PerDoc)
    local_prefill_tps: float = DEFAULT_LOCAL_PREFILL_TPS
    sources: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def writer(self, kind: str) -> PerDoc:
        return self.writer_by_kind.get(kind) or self.writer_all

    def teacher(self, kind: str) -> PerDoc:
        return self.teacher_by_kind.get(kind) or self.teacher_all


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def _rel(var_root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(var_root))
    except ValueError:
        return str(p)


_PASS_RE = re.compile(r"^wp5-(\d{4}-\d{2}-\d{2})\.(\d+)")


def pass_order(version: Any) -> tuple[str, int]:
    """A WP5 pass version as a sortable key: "wp5-2026-09-22.9" comes before
    "wp5-2026-09-22.10" (as text it would not); anything else sorts first."""
    m = _PASS_RE.match(str(version or ""))
    return (m.group(1), int(m.group(2))) if m else ("", -1)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _doc_states(dirs: Iterable[Path]) -> list[dict[str, Any]]:
    out = []
    for d in dirs:
        for p in sorted((d / "docs").glob("*.json")):
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def measure(var_root: Path, *, local_prefill_tps: float = DEFAULT_LOCAL_PREFILL_TPS) -> Measured:
    """Read every measured log named in the module docstring."""
    notes: list[str] = []
    sources: dict[str, list[str]] = {}

    writer_dirs = [var_root / d for d in WRITER_RUNS]
    writer_logs = _existing(d / "calls.jsonl" for d in writer_dirs)
    writer_rows = [r for p in writer_logs for r in load_calls(p, label=_rel(var_root, p))]
    sources["writer"] = [_rel(var_root, p) for p in writer_logs]

    planted = labeled = 0
    for d in writer_dirs:
        report = d / "run_report.json"
        if report.exists():
            for doc in json.loads(report.read_text(encoding="utf-8")).get("per_document", []):
                planted += int(doc.get("facts_planted") or 0)
                labeled += int(doc.get("facts_labeled") or 0)
    labeled_share = Rate(labeled / planted if planted else 0.0, planted)

    measure_dirs = sorted(
        Path(p) for p in glob.glob(str(var_root / TEACHER_MEASURE_GLOB), recursive=True)
    )
    teacher_logs = _existing(d / "teacher_calls.jsonl" for d in measure_dirs)
    teacher_rows = [
        r
        for p in teacher_logs
        for r in load_calls(p, label=_rel(var_root, p))
        if r.call_kind == "completeness" and r.attempt == 0
    ]
    sources["teacher"] = [_rel(var_root, p) for p in teacher_logs]

    first_pass = _doc_states(measure_dirs)
    regen = sum(1 for d in first_pass if d.get("status") == "needs_regeneration")
    regen_share = Rate(regen / len(first_pass) if first_pass else 0.0, len(first_pass))
    candidates = [
        sum(
            1
            for f in (d.get("attempts") or [{}])[0].get("facts", [])
            if f.get("role") == "completeness" and f.get("triage") == "candidate"
        )
        for d in first_pass
    ]
    cand_rate = Rate(sum(candidates) / len(candidates) if candidates else 0.0, len(candidates))

    wp5_dirs = sorted(
        Path(p) for p in glob.glob(str(var_root / WP5_SMOKE_GLOB)) if Path(p).is_dir()
    )
    wp5_logs = _existing(d / "teacher_calls.jsonl" for d in wp5_dirs)
    audit_rows = [
        r
        for p in wp5_logs
        for r in load_calls(p, label=_rel(var_root, p))
        if r.call_kind == "audit"
    ]
    sources["audit"] = [_rel(var_root, p) for p in wp5_logs]
    regenerated = [d for d in _doc_states(wp5_dirs) if len(d.get("attempts") or []) >= 2]
    kept = sum(1 for d in regenerated if d.get("status") in KEPT_STATUSES)
    kept_after = Rate(kept / len(regenerated) if regenerated else 0.0, len(regenerated))
    notes.append(
        "WP5 audit cost and the share kept after a regeneration come from the WP5 smoke, "
        "built on WP4 at 6606c6f (an older writer prompt)"
    )

    current_dirs = [
        d
        for d in [
            *wp5_dirs,
            *sorted(Path(p) for p in glob.glob(str(var_root / WP5_PILOT_PREP_GLOB))),
        ]
        if pass_order(_read_json(d / "qa_report.json").get("pass_version"))
        >= pass_order(CURRENT_POLICY_PASS)
    ]
    processed = [d for d in _doc_states(current_dirs) if d.get("attempts")]
    regen_now = sum(1 for d in processed if d.get("regeneration_tried"))
    regen_current = Rate(regen_now / len(processed) if processed else 0.0, len(processed))
    sources["wp5_current_rule"] = [_rel(var_root, d) for d in current_dirs]
    judge_dirs = [d for d in current_dirs if (d / "label_judge.jsonl").exists()]
    judge_rows = [
        row
        for d in judge_dirs
        for row in _read_jsonl(d / "label_judge.jsonl")
        if not row.get("error")
    ]
    judged_docs = sum(1 for d in _doc_states(judge_dirs) if d.get("attempts"))
    stamps = sorted(t for t in (_parse_at(r.get("created")) for r in judge_rows) if t is not None)
    judge_gaps = [
        (b - a).total_seconds()
        for a, b in zip(stamps, stamps[1:], strict=False)
        if 0 < (b - a).total_seconds() <= PAUSE_S
    ]
    label_judge = PerDoc(
        n=judged_docs,
        cost_usd=sum(float(r.get("cost_usd") or 0.0) for r in judge_rows) / (judged_docs or 1),
        calls=len(judge_rows) / (judged_docs or 1),
        seconds=(statistics.median(judge_gaps) if judge_gaps else 0.0)
        * len(judge_rows)
        / (judged_docs or 1),
    )
    sources["label_judge"] = [_rel(var_root, d / "label_judge.jsonl") for d in judge_dirs]

    neardup_path = var_root / NEARDUP_LOG
    neardup_rows = (
        load_calls(neardup_path, label=_rel(var_root, neardup_path))
        if neardup_path.exists()
        else []
    )
    sources["neardup"] = [_rel(var_root, neardup_path)] if neardup_rows else []

    judge_path = var_root / JUDGE_LOG
    judge = PerDoc()
    if judge_path.exists():
        disputes = [
            json.loads(line)
            for line in judge_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        costs = [sum(float(v.get("cost_usd") or 0) for v in d.get("votes", [])) for d in disputes]
        stamps = sorted(t for t in (_parse_at(d.get("created")) for d in disputes) if t is not None)
        gaps = [
            (b - a).total_seconds()
            for a, b in zip(stamps, stamps[1:], strict=False)
            if 0 < (b - a).total_seconds() <= PAUSE_S
        ]
        judge = PerDoc(
            n=len(disputes),
            cost_usd=sum(costs) / len(costs) if costs else 0.0,
            calls=sum(len(d.get("votes", [])) for d in disputes) / (len(disputes) or 1),
            seconds=statistics.median(gaps) if gaps else UNMEASURED_JUDGE_SECONDS_PER_DISPUTE,
        )
        if not gaps:
            notes.append(
                "judge seconds per dispute are not measured (one timestamp for every dispute); "
                f"{UNMEASURED_JUDGE_SECONDS_PER_DISPUTE:.0f} s from WP9's build note"
            )
        sources["judges"] = [_rel(var_root, judge_path)]

    out_tokens = seconds = 0.0
    mlx_logs = sorted(Path(p) for p in glob.glob(str(var_root / MLX_LOG_GLOB)))
    for p in mlx_logs:
        for m in _MLX_RE.finditer(p.read_text(encoding="utf-8", errors="replace")):
            out_tokens += int(m.group(3))
            seconds += float(m.group(4))
    decode = Rate(out_tokens / seconds if seconds else 0.0, int(out_tokens))
    sources["local_mlx"] = [_rel(var_root, p) for p in mlx_logs]

    return Measured(
        writer_by_kind=per_doc_by_kind(writer_rows),
        writer_all=per_doc(writer_rows),
        teacher_by_kind=per_doc_by_kind(teacher_rows),
        teacher_all=per_doc(teacher_rows),
        audit=per_doc(audit_rows),
        neardup=per_doc(neardup_rows),
        judge_per_dispute=judge,
        wp4_labeled_share=labeled_share,
        wp5_regeneration_share=regen_share,
        wp5_kept_after_regeneration=kept_after,
        wp5_candidates_per_doc=cand_rate,
        local_decode_tps=decode,
        local_prefill_tps=local_prefill_tps,
        wp5_regeneration_share_current=regen_current,
        label_judge=label_judge,
        sources=sources,
        notes=notes,
    )


# --------------------------------------------------------------------------
# World counts (deterministic, no calls)
# --------------------------------------------------------------------------


@dataclass
class WorldCounts:
    spec: WorldSpec
    docs_by_kind: dict[str, int]  # WP4 writes these (every role but near-duplicate)
    neardup_docs: int
    planted_facts: int  # scripted facts in WP4-written documents
    crm_code_labels: int


def world_counts(spec: WorldSpec) -> WorldCounts:
    from corpus.generator.v2.packs.schema import load_pack
    from corpus.generator.v2.simulator import simulate_world

    pack = load_pack(PACKS_DIR / f"{spec.pack}.yaml")
    world = simulate_world(pack, seed=spec.seed, months=spec.months, split=spec.split)
    plan = plan_world(world, pack)
    written = [d for d in plan.doc_plans if d.role != DocRole.NEAR_DUPLICATE]
    return WorldCounts(
        spec=spec,
        docs_by_kind=dict(Counter(d.kind.value for d in written)),
        neardup_docs=sum(1 for d in plan.doc_plans if d.role == DocRole.NEAR_DUPLICATE),
        planted_facts=sum(len(d.fact_ids) for d in written),
        crm_code_labels=len(plan.labels),
    )


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


@dataclass
class WorldEstimate:
    world: str
    strictness: str
    writers_local: bool
    documents: int
    cost_usd: dict[str, float]
    seconds: dict[str, float]
    labeled_facts: float
    unmeasured: list[str] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(self.cost_usd.values())

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds.values())

    @property
    def usd_per_labeled_fact(self) -> float | None:
        return self.total_usd / self.labeled_facts if self.labeled_facts else None


def local_seconds(doc: PerDoc, m: Measured) -> float:
    """One document's writer calls on a local model: prompt at the assumed
    prefill speed, completion at the measured decode speed."""
    decode = m.local_decode_tps.value or 1.0
    return doc.prompt_tokens / m.local_prefill_tps + doc.completion_tokens / decode


def estimate_world(
    counts: WorldCounts, m: Measured, *, writers_local: bool = False
) -> WorldEstimate:
    strict = counts.spec.strictness == "test"
    cost: dict[str, float] = defaultdict(float)
    secs: dict[str, float] = defaultdict(float)
    unmeasured: list[str] = []
    p_regen = m.wp5_regeneration_share_current.value
    docs = sum(counts.docs_by_kind.values())

    def writer_cost_secs(doc: PerDoc) -> tuple[float, float]:
        if writers_local:
            return 0.0, local_seconds(doc, m)
        return doc.cost_usd, doc.seconds

    for kind, n in counts.docs_by_kind.items():
        w_cost, w_secs = writer_cost_secs(m.writer(kind))
        t = m.teacher(kind)
        cost["write (WP4)"] += n * w_cost
        secs["write (WP4)"] += n * w_secs
        cost["completeness teacher (WP5)"] += n * t.cost_usd
        secs["completeness teacher (WP5)"] += n * t.seconds
        cost["completeness regeneration (WP5)"] += n * p_regen * (w_cost + t.cost_usd)
        secs["completeness regeneration (WP5)"] += n * p_regen * (w_secs + t.seconds)

    # WP6 writes the near-duplicates after WP5, so no teacher reads them.
    nd_cost, nd_secs = writer_cost_secs(m.neardup)
    cost["near-duplicates (WP6)"] += counts.neardup_docs * nd_cost
    secs["near-duplicates (WP6)"] += counts.neardup_docs * nd_secs

    all_docs = docs + counts.neardup_docs
    audit_share = 1.0 if strict else AUDIT_SHARE
    cost["audit teacher (WP5)"] += docs * audit_share * m.audit.cost_usd
    secs["audit teacher (WP5)"] += docs * audit_share * m.audit.seconds
    cost["label judge (WP5)"] += docs * m.label_judge.cost_usd
    secs["label judge (WP5)"] += docs * m.label_judge.seconds

    if strict:
        disputes = all_docs * m.wp5_candidates_per_doc.value
        cost["three-judge disputes (test)"] += disputes * m.judge_per_dispute.cost_usd
        secs["three-judge disputes (test)"] += disputes * m.judge_per_dispute.seconds
        cost["questions (WP10, spec figure)"] += SPEC_QUESTIONS_USD_PER_TEST_WORLD
        unmeasured.append(
            "questions: WP10 not built; the spec's $0.86 is carried, time not counted"
        )

    kept = (1 - p_regen) + p_regen * m.wp5_kept_after_regeneration.value
    labeled = counts.planted_facts * m.wp4_labeled_share.value * kept
    return WorldEstimate(
        world=counts.spec.world_id,
        strictness=counts.spec.strictness,
        writers_local=writers_local,
        documents=all_docs,
        cost_usd=dict(cost),
        seconds=dict(secs),
        labeled_facts=labeled,
        unmeasured=unmeasured,
    )


@dataclass
class RoundEstimate:
    worlds: list[WorldEstimate]

    @property
    def total_usd(self) -> float:
        return sum(w.total_usd for w in self.worlds)

    @property
    def total_seconds(self) -> float:
        return sum(w.total_seconds for w in self.worlds)

    @property
    def labeled_facts(self) -> float:
        return sum(w.labeled_facts for w in self.worlds)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _usd(x: float) -> str:
    return f"${x:,.2f}"


def _hours(s: float) -> str:
    return f"{s / 3600:,.1f} h"


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def render_markdown(
    m: Measured, paid: RoundEstimate, local: RoundEstimate, counts: Sequence[WorldCounts]
) -> str:
    lines: list[str] = []

    def world_rows(est: RoundEstimate, strict: bool) -> list[WorldEstimate]:
        return [w for w in est.worlds if (w.strictness == "test") == strict]

    lines.append("## v2 cost model (measured)\n")
    lines.append(
        "| Writers | Per train/validation world | Per test world | Round "
        f"({len(paid.worlds)} worlds) | Labeled facts, round | $ per labeled fact | "
        "Wall time per train world (c=1) | Round wall time (c=1) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, est in (("paid rotation", paid), ("local at $0", local)):
        train = world_rows(est, False)
        test = world_rows(est, True)
        per_fact = est.total_usd / est.labeled_facts if est.labeled_facts else 0.0
        lines.append(
            f"| {name} | {_usd(_mean([w.total_usd for w in train]))} | "
            f"{_usd(_mean([w.total_usd for w in test]))} | {_usd(est.total_usd)} | "
            f"{est.labeled_facts:,.0f} | {_usd(per_fact)} | "
            f"{_hours(_mean([w.total_seconds for w in train]))} | {_hours(est.total_seconds)} |"
        )
    lines.append(
        f"\nSpec section 8 modeled round 1 at {_usd(SPEC_ROUND1_USD)} "
        f"({_usd(SPEC_ROUND1_USD_MEASURED_TEACHER)} at the measured teacher price), "
        f"{_usd(SPEC_TRAIN_WORLD_USD)} per train world and {_usd(SPEC_TEST_WORLD_USD)} per test world.\n"
    )

    lines.append(
        "### Per stage, one train world, paid writers (mean over the round's train and validation worlds)\n"
    )
    lines.append("| Stage | Cost | Wall time (c=1) |")
    lines.append("|---|---|---|")
    train = world_rows(paid, False)
    stages = sorted({k for w in train for k in w.cost_usd})
    for stage in stages:
        lines.append(
            f"| {stage} | {_usd(_mean([w.cost_usd.get(stage, 0.0) for w in train]))} | "
            f"{_hours(_mean([w.seconds.get(stage, 0.0) for w in train]))} |"
        )

    lines.append("\n### Measured inputs\n")
    lines.append("| Measure | Kind | n | Cost per doc | Calls per doc | Seconds per doc |")
    lines.append("|---|---|---|---|---|---|")
    for label, table, overall in (
        ("writer (WP4)", m.writer_by_kind, m.writer_all),
        ("teacher (WP5 Kimi K2)", m.teacher_by_kind, m.teacher_all),
    ):
        for kind, d in sorted(table.items()):
            lines.append(
                f"| {label} | {kind} | {d.n} | ${d.cost_usd:.5f} | {d.calls:.2f} | {d.seconds:.0f} |"
            )
        lines.append(
            f"| {label} | all | {overall.n} | ${overall.cost_usd:.5f} | {overall.calls:.2f} | {overall.seconds:.0f} |"
        )
    for label, d in (
        ("audit teacher (WP5)", m.audit),
        ("label judge (WP5)", m.label_judge),
        ("near-duplicate writer (WP6)", m.neardup),
        ("three judges, per dispute (WP9)", m.judge_per_dispute),
    ):
        lines.append(
            f"| {label} | all | {d.n} | ${d.cost_usd:.5f} | {d.calls:.2f} | {d.seconds:.0f} |"
        )
    lines.append("")
    lines.append("| Rate | Value | Measured on |")
    lines.append("|---|---|---|")
    for label, rate, unit in (
        ("WP4 labeled share of planted facts", m.wp4_labeled_share, "planted facts"),
        (
            "WP5 documents regenerated, current rule (used)",
            m.wp5_regeneration_share_current,
            "documents",
        ),
        (
            "WP5 documents needing regeneration, section 3.4 rule (not used)",
            m.wp5_regeneration_share,
            "documents",
        ),
        (
            "WP5 documents kept after a regeneration",
            m.wp5_kept_after_regeneration,
            "regenerated documents",
        ),
        (
            "WP5 candidates per document (test-world disputes)",
            m.wp5_candidates_per_doc,
            "documents",
        ),
        ("local MLX decode, tokens per second", m.local_decode_tps, "output tokens"),
    ):
        lines.append(f"| {label} | {rate.value:.3f} | {rate.n} {unit} |")
    lines.append(
        f"| local prefill, tokens per second | {m.local_prefill_tps:.0f} | **assumed, not measured** |"
    )

    total_docs = sum(sum(c.docs_by_kind.values()) + c.neardup_docs for c in counts)
    lines.append(
        f"\nWorld sizes from simulate and plan: {len(counts)} worlds, {total_docs:,} documents, "
        f"{sum(c.planted_facts for c in counts):,} planted facts, "
        f"{sum(c.crm_code_labels for c in counts):,} CRM code labels (free, not in the fact count).\n"
    )
    lines.append("### Not measured or assumed\n")
    flags = [
        "labeled facts = planted x WP4 labeled share x WP5 keep rate; silver labels not counted",
        "local writers are assumed to need as many repairs and regenerations as the paid rotation",
        "wall time is at concurrency 1: the runner writes one document at a time, one world at a time",
        *m.notes,
        *sorted({u for w in paid.worlds for u in w.unmeasured}),
    ]
    lines.extend(f"- {f}" for f in flags)
    return "\n".join(lines) + "\n"


def to_json(m: Measured, paid: RoundEstimate, local: RoundEstimate) -> dict[str, Any]:
    def est(e: RoundEstimate) -> dict[str, Any]:
        return {
            "total_usd": round(e.total_usd, 4),
            "total_hours": round(e.total_seconds / 3600, 2),
            "labeled_facts": round(e.labeled_facts, 1),
            "worlds": [
                {
                    "world": w.world,
                    "strictness": w.strictness,
                    "documents": w.documents,
                    "total_usd": round(w.total_usd, 4),
                    "hours": round(w.total_seconds / 3600, 3),
                    "labeled_facts": round(w.labeled_facts, 1),
                    "cost_usd": {k: round(v, 5) for k, v in w.cost_usd.items()},
                    "seconds": {k: round(v, 1) for k, v in w.seconds.items()},
                    "unmeasured": w.unmeasured,
                }
                for w in e.worlds
            ],
        }

    return {"measured": asdict(m), "paid_writers": est(paid), "local_writers": est(local)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/data/v2_estimate.py", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--var-root", type=Path, default=REPO_ROOT / "var")
    parser.add_argument(
        "--worlds", default="round1", help="as synth_v2.py --worlds (default round1)"
    )
    parser.add_argument("--pack", default="pharmacy_services")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--local-prefill-tps", type=float, default=DEFAULT_LOCAL_PREFILL_TPS)
    parser.add_argument("--json", type=Path, default=None, help="also write the full estimate here")
    args = parser.parse_args(argv)

    m = measure(args.var_root, local_prefill_tps=args.local_prefill_tps)
    if m.writer_all.n == 0 or m.teacher_all.n == 0:
        print(f"error: no writer or teacher call logs under {args.var_root}", file=sys.stderr)
        return 2
    specs = parse_worlds(args.worlds, pack=args.pack, split=args.split, seed=args.seed)
    counts = [world_counts(s) for s in specs]
    paid = RoundEstimate([estimate_world(c, m) for c in counts])
    local = RoundEstimate([estimate_world(c, m, writers_local=True) for c in counts])
    print(render_markdown(m, paid, local, counts))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(to_json(m, paid, local), indent=2, default=str), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
