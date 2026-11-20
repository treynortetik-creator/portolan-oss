#!/usr/bin/env python3
"""Miss analysis for the 2026-09-18-corrected extraction bench.

Reads the saved, scored bench output for Kimi K2 and Nova Lite (three runs
each, `var/bench/2026-09-18-corrected/<model-slug>/run{1,2,3}/bench-
2026-09-18-openrouter.json`) plus the truth table (`corpus/truth-table.yaml`
+ `corpus/out/manifest.json`, joined via `portolan.bench.truth.Truth`) and
answers: which of the 73 planted facts does Kimi K2 miss, how consistently,
why (per `portolan.bench.extraction`'s own matcher: type / entities /
locator / claim-similarity), and what would recall look like if the
matcher's near-misses were credited.

No model calls, no network. Every number this script prints is read
straight from the saved JSON or computed from the corpus text on disk --
nothing hand-typed into the report this script's output feeds.

Usage:
    uv run python scripts/bench_miss_analysis.py [--bench-root PATH] [--json OUT]

Prints a full human-readable report to stdout. `--json` additionally dumps
every intermediate table (fact status matrix, per-fact classification,
breakdowns, recall-ceiling numbers) as JSON for a second pass without
re-deriving anything.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from portolan.bench.corpus_loader import load_prepared_source  # noqa: E402
from portolan.bench.locators import parse_hhmmss, trailing_int  # noqa: E402
from portolan.bench.truth import Truth  # noqa: E402

DEFAULT_BENCH_ROOT = Path(
    "var/bench/2026-09-18-corrected"
)
DEFAULT_TRUTH_TABLE = REPO_ROOT / "corpus" / "truth-table.yaml"
DEFAULT_MANIFEST = REPO_ROOT / "corpus" / "out" / "manifest.json"
DEFAULT_CORPUS_ROOT = REPO_ROOT / "corpus" / "out"

MODEL_SLUGS = {"kimi": "moonshotai_kimi-k2", "nova": "amazon_nova-lite-v1"}
RUNS = ["run1", "run2", "run3"]
REPORT_FILENAME = "bench-2026-09-18-openrouter.json"

# Below this best_similarity, a run's "closest candidate" is judged to not
# really be an attempt at the planted fact at all (just whichever produced
# fact from that source happened to have the fewest failed criteria) --
# see the histogram this script prints and the write-up's own reasoning
# for where this sits relative to the two thresholds the scorer itself
# already draws (_CLAIM_SIMILARITY_THRESHOLD = 0.30, _FUZZY_QUOTE_THRESHOLD
# = 0.75). Kept as a named constant so the report can quote it directly.
NOT_EXTRACTED_SIMILARITY_FLOOR = 0.15

_TIMESTAMP_RE = re.compile(r"\[(\d{1,2}):(\d{2}):(\d{2})\]")
_TICKET_LOCATOR_RE = re.compile(r"#(\d+)\]")
_EMAIL_PARA_RE = re.compile(r"\[\xb6(\d+)\]")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_truth() -> Truth:
    return Truth.load(DEFAULT_TRUTH_TABLE, DEFAULT_MANIFEST)


def load_run(bench_root: Path, model_slug: str, run: str) -> dict[str, Any]:
    path = bench_root / model_slug / run / REPORT_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["extraction"]


def load_model_runs(bench_root: Path, model_key: str) -> list[dict[str, Any]]:
    slug = MODEL_SLUGS[model_key]
    return [load_run(bench_root, slug, run) for run in RUNS]


# --------------------------------------------------------------------------
# source text stats: word count + locator position as a fraction of doc
# --------------------------------------------------------------------------


class SourceStats:
    """Memoized word count + position-fraction resolver over the exact
    prepared text `run_extraction` fed the backend (same `corpus_loader`
    the bench itself uses), keyed by source_id."""

    def __init__(self, truth: Truth, corpus_root: Path) -> None:
        self._truth = truth
        self._corpus_root = corpus_root
        self._text_cache: dict[str, str] = {}
        self._wordcount_cache: dict[str, int] = {}

    def text(self, source_id: str) -> str:
        if source_id not in self._text_cache:
            manifest_source = self._truth.manifest.source(source_id)
            prepared = load_prepared_source(manifest_source, corpus_root=self._corpus_root)
            self._text_cache[source_id] = prepared.text
        return self._text_cache[source_id]

    def word_count(self, source_id: str) -> int:
        if source_id not in self._wordcount_cache:
            self._wordcount_cache[source_id] = len(self.text(source_id).split())
        return self._wordcount_cache[source_id]

    def position(self, source_id: str, source_type: str, locator: str) -> float | None:
        """Fraction (0.0 = start, 1.0 = end) of the document the locator
        sits at, by the same unit the source's own locator window uses
        (call: seconds into the transcript; ticket: comment number; email:
        paragraph number). None for rss/crm -- those are single-row
        sources with no "position within the document" to speak of."""
        text = self.text(source_id)
        if source_type == "call":
            point = parse_hhmmss(locator)
            all_ts = [
                int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in _TIMESTAMP_RE.findall(text)
            ]
            if point is None or not all_ts or max(all_ts) == 0:
                return None
            return min(point / max(all_ts), 1.0)
        if source_type == "ticket":
            point = trailing_int(locator)
            all_n = [int(n) for n in _TICKET_LOCATOR_RE.findall(text)]
            if point is None or not all_n or max(all_n) == 0:
                return None
            return min(point / max(all_n), 1.0)
        if source_type == "email":
            point = trailing_int(locator)
            all_n = [int(n) for n in _EMAIL_PARA_RE.findall(text)]
            if point is None or not all_n or max(all_n) == 0:
                return None
            return min(point / max(all_n), 1.0)
        return None  # rss / crm: single row, no meaningful position


# --------------------------------------------------------------------------
# fact status matrix: matched / missed / not_attempted per fact per run
# --------------------------------------------------------------------------


def fact_status_matrix(truth: Truth, runs_data: list[dict[str, Any]]) -> dict[str, list[str]]:
    """{fact_id: [status_run1, status_run2, status_run3]}, status in
    {"matched", "missed", "not_attempted"}. "not_attempted" only when
    EVERY source this fact is planted in failed extraction in that run
    (mirrors how `score_extraction`'s own `planted_total` denominator
    drops a fact whose only source failed)."""
    matched_sets = [{m["fact_id"] for m in run["matched"]} for run in runs_data]
    failed_sets = [set(run["failed_sources"].keys()) for run in runs_data]

    matrix: dict[str, list[str]] = {}
    for fact_id in truth.planted_fact_ids():
        fact_sources = [ref.id for ref in truth.fact(fact_id).sources]
        statuses = []
        for i in range(3):
            if fact_id in matched_sets[i]:
                statuses.append("matched")
            else:
                attempted = any(s not in failed_sets[i] for s in fact_sources)
                statuses.append("missed" if attempted else "not_attempted")
        matrix[fact_id] = statuses
    return matrix


def bucket_facts(matrix: dict[str, list[str]]) -> dict[str, list[str]]:
    """fact ids bucketed by how many of the 3 runs matched them, restricted
    to facts attempted in all 3 runs (the clean case -- every fact in this
    corpus's Kimi K2 runs; Nova Lite has one exception, reported
    separately)."""
    buckets: dict[str, list[str]] = {
        "missed_all3": [],
        "missed_2of3": [],
        "missed_1of3": [],
        "never_missed": [],
        "partial_data": [],  # not attempted in all 3 runs -- reported separately
    }
    for fact_id, statuses in matrix.items():
        if "not_attempted" in statuses:
            buckets["partial_data"].append(fact_id)
            continue
        matched_count = statuses.count("matched")
        if matched_count == 0:
            buckets["missed_all3"].append(fact_id)
        elif matched_count == 1:
            buckets["missed_2of3"].append(fact_id)
        elif matched_count == 2:
            buckets["missed_1of3"].append(fact_id)
        else:
            buckets["never_missed"].append(fact_id)
    return buckets


# --------------------------------------------------------------------------
# near-miss classification
# --------------------------------------------------------------------------


def unmatched_entries_for(runs_data: list[dict[str, Any]], fact_id: str) -> list[dict[str, Any]]:
    """Every UnmatchedFact record (across all 3 runs, and across every
    source the fact is planted in) for this fact_id."""
    out = []
    for run in runs_data:
        for u in run["unmatched"]:
            if u["fact_id"] == fact_id:
                out.append(u)
    return out


def classify_entry(entry: dict[str, Any]) -> str:
    """One UnmatchedFact record -> one of the four classifications."""
    candidate = entry.get("best_candidate")
    similarity = entry.get("best_similarity", 0.0)
    if candidate is None or similarity < NOT_EXTRACTED_SIMILARITY_FLOOR:
        return "not_extracted"
    failed = set(entry.get("failed_criteria", []))
    if failed == {"claim_similarity"}:
        return "paraphrased_below_threshold"
    if failed == {"type"}:
        return "mis_typed"
    if failed and failed <= {"entities", "locator"}:
        return "entity_locator_mismatch"
    # Mixed failures (e.g. type + claim_similarity together): still
    # "extracted something", but no single clean reason -- bucketed by the
    # criterion most likely to be the primary blocker, checked in a fixed
    # priority order so every entry lands in exactly one bucket.
    if "type" in failed:
        return "mis_typed"
    if "claim_similarity" in failed:
        return "paraphrased_below_threshold"
    if failed & {"entities", "locator"}:
        return "entity_locator_mismatch"
    return "not_extracted"


def classify_fact(runs_data: list[dict[str, Any]], fact_id: str) -> dict[str, Any]:
    """Classification for one consistently-missed fact: take every
    UnmatchedFact record across all 3 runs and all its sources, pick the
    one with the highest best_similarity (the closest the model ever got
    across every attempt), classify that one, and return it plus the full
    list for the report to show every attempt if useful."""
    entries = unmatched_entries_for(runs_data, fact_id)
    if not entries:
        return {
            "classification": "no_unmatched_record",
            "best_entry": None,
            "all_entries": [],
        }
    best_entry = max(entries, key=lambda e: e.get("best_similarity", 0.0))
    return {
        "classification": classify_entry(best_entry),
        "best_entry": best_entry,
        "all_entries": entries,
    }


# --------------------------------------------------------------------------
# recall ceiling: what if near-misses were credited?
# --------------------------------------------------------------------------


def recall_ceiling_per_run(run: dict[str, Any]) -> dict[str, Any]:
    """Actual vs. ceiling recall for one run, crediting every unmatched
    fact whose best candidate classifies as anything other than
    'not_extracted' (i.e. the matcher's near-misses), deduped by fact_id
    since a multi-source fact can carry two UnmatchedFact records in one
    run."""
    matched_ids = {m["fact_id"] for m in run["matched"]}
    near_miss_ids: set[str] = set()
    for u in run["unmatched"]:
        if u["fact_id"] in matched_ids:
            continue
        if classify_entry(u) != "not_extracted":
            near_miss_ids.add(u["fact_id"])
    planted_total = run["planted_total"]
    actual_recall = run["recall"]
    ceiling_matched = len(matched_ids) + len(near_miss_ids)
    ceiling_recall = ceiling_matched / planted_total if planted_total else 0.0
    missed_count = planted_total - len(matched_ids)
    near_miss_share_of_misses = len(near_miss_ids) / missed_count if missed_count else None
    return {
        "planted_total": planted_total,
        "matched_count": len(matched_ids),
        "near_miss_fact_ids": sorted(near_miss_ids),
        "ceiling_matched": ceiling_matched,
        "actual_recall": actual_recall,
        "ceiling_recall": ceiling_recall,
        "missed_count": missed_count,
        "near_miss_share_of_misses": near_miss_share_of_misses,
    }


# --------------------------------------------------------------------------
# breakdowns over a set of consistent misses
# --------------------------------------------------------------------------


def position_bucket(pos: float | None) -> str:
    if pos is None:
        return "n/a (single-row source)"
    if pos < 1 / 3:
        return "early (0-33%)"
    if pos < 2 / 3:
        return "middle (33-66%)"
    return "late (66-100%)"


def build_fact_detail(
    truth: Truth, stats: SourceStats, runs_data: list[dict[str, Any]], fact_id: str
) -> dict[str, Any]:
    tf = truth.fact(fact_id)
    account_ids = {a.id for a in truth.table.accounts}
    account = next((e for e in tf.entities if e in account_ids), None)
    sources = []
    for ref in tf.sources:
        wc = stats.word_count(ref.id)
        # Use the fact's own primary locator for THIS source, from the
        # joined PlantedFact (truth.planted(source_id) picks it out).
        planted_here = next((pf for pf in truth.planted(ref.id) if pf.fact_id == fact_id), None)
        locator = planted_here.locator if planted_here else None
        pos = stats.position(ref.id, ref.type, locator) if locator else None
        sources.append(
            {
                "source_id": ref.id,
                "source_type": ref.type,
                "word_count": wc,
                "locator": locator,
                "position": pos,
                "position_bucket": position_bucket(pos),
            }
        )
    classification = classify_fact(runs_data, fact_id)
    return {
        "fact_id": fact_id,
        "fact_type": tf.type,
        "claim": tf.claim,
        "account": account,
        "tests": list(tf.tests),
        "sources": sources,
        "classification": classification["classification"],
        "best_entry": classification["best_entry"],
        "all_entries": classification["all_entries"],
    }


def counter_table(counter: Counter) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))


# --------------------------------------------------------------------------
# report rendering
# --------------------------------------------------------------------------


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def render_report(
    truth: Truth,
    stats: SourceStats,
    kimi_runs: list[dict[str, Any]],
    nova_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    dump: dict[str, Any] = {}

    kimi_matrix = fact_status_matrix(truth, kimi_runs)
    kimi_buckets = bucket_facts(kimi_matrix)
    dump["kimi_matrix"] = kimi_matrix
    dump["kimi_buckets"] = kimi_buckets

    print_header("1. Kimi K2 -- miss consistency across 3 runs")
    print(f"Planted facts: {len(kimi_matrix)}")
    print(f"Missed in ALL 3 runs (consistent miss): {len(kimi_buckets['missed_all3'])}")
    print(f"Missed in 2 of 3 runs (found in exactly 1): {len(kimi_buckets['missed_2of3'])}")
    print(f"Missed in 1 of 3 runs (found in exactly 2): {len(kimi_buckets['missed_1of3'])}")
    print(f"Never missed (found in all 3): {len(kimi_buckets['never_missed'])}")
    print(
        f"Partial data (a source failed extraction in >=1 run): {len(kimi_buckets['partial_data'])}"
    )

    def fact_detail_rows(fact_ids: list[str]) -> list[dict[str, Any]]:
        return [
            build_fact_detail(truth, stats, kimi_runs, fid)
            for fid in sorted(fact_ids, key=lambda f: (len(f), f))
        ]

    all3 = fact_detail_rows(kimi_buckets["missed_all3"])
    twoof3 = fact_detail_rows(kimi_buckets["missed_2of3"])
    oneof3 = fact_detail_rows(kimi_buckets["missed_1of3"])
    dump["kimi_missed_all3_detail"] = all3
    dump["kimi_missed_2of3_detail"] = twoof3
    dump["kimi_missed_1of3_detail"] = oneof3

    def print_fact_row(d: dict[str, Any]) -> None:
        srcs = "; ".join(f"{s['source_id']}({s['source_type']})" for s in d["sources"])
        print(
            f"  {d['fact_id']:<6} {d['fact_type']:<18} acct={d['account'] or '-':<18} "
            f"src=[{srcs}] tests={d['tests']}"
        )

    print("\n-- Missed in all 3 runs --")
    for d in all3:
        print_fact_row(d)
    print("\n-- Missed in 2 of 3 runs --")
    for d in twoof3:
        print_fact_row(d)
    print("\n-- Missed in 1 of 3 runs --")
    for d in oneof3:
        print_fact_row(d)

    if kimi_buckets["partial_data"]:
        print("\n-- Partial data (a source failed to extract in >=1 run) --")
        for fid in kimi_buckets["partial_data"]:
            print(f"  {fid}: statuses={kimi_matrix[fid]}")

    # -- 2. breakdowns over the consistent-miss set -------------------------
    print_header("2. Consistent-miss (missed all 3 runs) breakdowns -- Kimi K2")
    by_type = Counter(d["fact_type"] for d in all3)
    by_source_type = Counter(s["source_type"] for d in all3 for s in d["sources"])
    by_position = Counter(s["position_bucket"] for d in all3 for s in d["sources"])
    by_tests = Counter(t for d in all3 for t in d["tests"])

    print("\nBy fact type:")
    for k, v in counter_table(by_type):
        print(f"  {k:<20} {v}")

    print("\nBy source type (source-instance level -- a multi-source fact counts once per source):")
    for k, v in counter_table(by_source_type):
        print(f"  {k:<10} {v}")

    print("\nWord count of the source each consistent miss sits in:")
    wcs = [s["word_count"] for d in all3 for s in d["sources"]]
    if wcs:
        print(
            f"  n={len(wcs)} min={min(wcs)} median={statistics.median(wcs):.0f} max={max(wcs)} mean={statistics.mean(wcs):.0f}"
        )
    # Compare against the corpus-wide word count of every call/ticket/email
    # source that carries ANY planted fact (not just misses), so "do misses
    # skew toward long sources" has a baseline to read against.
    all_planted_source_ids = {
        ref.id for f in truth.table.facts if not f.is_distractor for ref in f.sources
    }
    all_wc_by_type: dict[str, list[int]] = defaultdict(list)
    for sid in all_planted_source_ids:
        st = truth.manifest.source(sid).type
        all_wc_by_type[st].append(stats.word_count(sid))
    print(
        "\n  Corpus-wide word count, by source type, over every source carrying >=1 planted fact:"
    )
    for st, wc_list in sorted(all_wc_by_type.items()):
        print(
            f"    {st:<8} n={len(wc_list)} median={statistics.median(wc_list):.0f} mean={statistics.mean(wc_list):.0f} max={max(wc_list)}"
        )
    miss_wc_by_type: dict[str, list[int]] = defaultdict(list)
    for d in all3:
        for s in d["sources"]:
            miss_wc_by_type[s["source_type"]].append(s["word_count"])
    print("\n  Word count, by source type, restricted to consistent-miss sources:")
    for st, wc_list in sorted(miss_wc_by_type.items()):
        print(
            f"    {st:<8} n={len(wc_list)} median={statistics.median(wc_list):.0f} mean={statistics.mean(wc_list):.0f} max={max(wc_list)}"
        )

    print("\nPosition within the source (fraction of document):")
    for k, v in counter_table(by_position):
        print(f"  {k:<25} {v}")

    print("\nBy truth-table `tests` tag:")
    for k, v in counter_table(by_tests):
        print(f"  {k:<35} {v}")

    dump["kimi_all3_by_type"] = dict(by_type)
    dump["kimi_all3_by_source_type"] = dict(by_source_type)
    dump["kimi_all3_by_position"] = dict(by_position)
    dump["kimi_all3_by_tests"] = dict(by_tests)
    dump["corpus_wide_wordcount_by_source_type"] = {
        k: {"n": len(v), "median": statistics.median(v), "mean": statistics.mean(v), "max": max(v)}
        for k, v in all_wc_by_type.items()
    }
    dump["miss_wordcount_by_source_type"] = {
        k: {"n": len(v), "median": statistics.median(v), "mean": statistics.mean(v), "max": max(v)}
        for k, v in miss_wc_by_type.items()
    }

    # -- 3. near-miss classification -----------------------------------------
    print_header("3. Near-miss classification -- consistent misses, Kimi K2")
    print(
        f"\nSimilarity floor for 'not_extracted' vs. a real candidate: best_similarity < {NOT_EXTRACTED_SIMILARITY_FLOOR}"
    )
    print(
        "Histogram of best_similarity across all UnmatchedFact records for consistent misses (all runs, all sources):"
    )
    all_best_sims = sorted(e.get("best_similarity", 0.0) for d in all3 for e in d["all_entries"])
    if all_best_sims:
        buckets_edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.01]
        hist = Counter()
        for v in all_best_sims:
            for lo, hi in pairwise(buckets_edges):
                if lo <= v < hi:
                    hist[f"[{lo:.2f},{hi:.2f})"] += 1
                    break
        for k in sorted(hist, key=lambda s: float(s.split(",")[0][1:])):
            print(f"  {k:<16} {hist[k]}")
    dump["kimi_all3_best_similarity_values"] = all_best_sims

    class_counts = Counter(d["classification"] for d in all3)
    print("\nClassification of consistent misses (by fact, using the best attempt across 3 runs):")
    for k, v in counter_table(class_counts):
        print(f"  {k:<28} {v}")
    total_all3 = len(all3)
    not_extracted = class_counts.get("not_extracted", 0)
    near_miss_count = total_all3 - not_extracted
    print(f"\nTotal consistent misses: {total_all3}")
    print(f"Real extraction misses (not_extracted): {not_extracted}")
    print(
        f"Near-misses (matcher-strictness misses): {near_miss_count} ({fmt_pct(near_miss_count / total_all3) if total_all3 else 'n/a'})"
    )
    dump["kimi_all3_classification_counts"] = dict(class_counts)
    dump["kimi_all3_near_miss_share"] = (near_miss_count / total_all3) if total_all3 else None

    print("\nPer-fact detail -- planted claim vs. nearest produced claim:")
    for d in all3:
        best = d["best_entry"]
        print(
            f"\n  {d['fact_id']} [{d['classification']}] ({d['fact_type']}, {d['sources'][0]['source_type'] if d['sources'] else '?'})"
        )
        print(f"    planted:  {d['claim']}")
        if best and best.get("best_candidate"):
            bc = best["best_candidate"]
            print(f"    nearest:  {bc['claim']}")
            print(
                f"      produced type={bc['type']} entities={bc['entities']} locator={bc['locator']} quote={bc.get('quote')!r}"
            )
            print(
                f"      failed_criteria={best['failed_criteria']} best_similarity={best['best_similarity']:.3f}"
            )
        else:
            print(
                "    nearest:  (no produced fact for this source came close -- source produced nothing usable)"
            )

    # -- 4. Nova Lite brief table --------------------------------------------
    print_header("4. Nova Lite -- brief comparison")
    nova_matrix = fact_status_matrix(truth, nova_runs)
    nova_buckets = bucket_facts(nova_matrix)
    dump["nova_matrix"] = nova_matrix
    dump["nova_buckets"] = nova_buckets
    print(f"Planted facts: {len(nova_matrix)}")
    print(f"Missed in ALL 3 runs (consistent miss): {len(nova_buckets['missed_all3'])}")
    print(f"Missed in 2 of 3: {len(nova_buckets['missed_2of3'])}")
    print(f"Missed in 1 of 3: {len(nova_buckets['missed_1of3'])}")
    print(f"Never missed: {len(nova_buckets['never_missed'])}")
    print(f"Partial data (source failed in >=1 run): {len(nova_buckets['partial_data'])}")
    if nova_buckets["partial_data"]:
        for fid in nova_buckets["partial_data"]:
            print(f"  {fid}: statuses={nova_matrix[fid]}")

    nova_all3 = fact_detail_rows_for(truth, stats, nova_runs, nova_buckets["missed_all3"])
    dump["nova_missed_all3_detail"] = nova_all3
    nova_by_type = Counter(d["fact_type"] for d in nova_all3)
    nova_by_source_type = Counter(s["source_type"] for d in nova_all3 for s in d["sources"])
    nova_class_counts = Counter(d["classification"] for d in nova_all3)
    print("\nNova Lite consistent misses -- by fact type:")
    for k, v in counter_table(nova_by_type):
        print(f"  {k:<20} {v}")
    print("\nNova Lite consistent misses -- by source type:")
    for k, v in counter_table(nova_by_source_type):
        print(f"  {k:<10} {v}")
    print("\nNova Lite consistent misses -- classification:")
    for k, v in counter_table(nova_class_counts):
        print(f"  {k:<28} {v}")
    dump["nova_all3_by_type"] = dict(nova_by_type)
    dump["nova_all3_by_source_type"] = dict(nova_by_source_type)
    dump["nova_all3_classification_counts"] = dict(nova_class_counts)

    kimi_all3_ids = set(kimi_buckets["missed_all3"])
    nova_all3_ids = set(nova_buckets["missed_all3"])
    both = sorted(kimi_all3_ids & nova_all3_ids)
    kimi_only = sorted(kimi_all3_ids - nova_all3_ids)
    nova_only = sorted(nova_all3_ids - kimi_all3_ids)
    print(f"\nOverlap of consistent misses: both models={len(both)} {both}")
    print(f"Kimi-only consistent misses: {len(kimi_only)} {kimi_only}")
    print(f"Nova-only consistent misses: {len(nova_only)} {nova_only}")
    dump["overlap_both"] = both
    dump["overlap_kimi_only"] = kimi_only
    dump["overlap_nova_only"] = nova_only

    # -- 5. recall ceiling ----------------------------------------------------
    print_header("5. Recall ceiling if near-misses were credited")
    for label, runs_data in (("Kimi K2", kimi_runs), ("Nova Lite", nova_runs)):
        print(f"\n{label}:")
        ceilings = []
        for i, run in enumerate(runs_data, start=1):
            c = recall_ceiling_per_run(run)
            ceilings.append(c)
            print(
                f"  run{i}: actual={fmt_pct(c['actual_recall'])} "
                f"({c['matched_count']}/{c['planted_total']}) -> "
                f"ceiling={fmt_pct(c['ceiling_recall'])} "
                f"({c['ceiling_matched']}/{c['planted_total']}, "
                f"+{len(c['near_miss_fact_ids'])} near-miss facts credited) | "
                f"of {c['missed_count']} facts missed this run, "
                f"{len(c['near_miss_fact_ids'])} ({fmt_pct(c['near_miss_share_of_misses'])}) were near-misses"
            )
        actuals = [c["actual_recall"] for c in ceilings]
        ceils = [c["ceiling_recall"] for c in ceilings]
        shares = [
            c["near_miss_share_of_misses"]
            for c in ceilings
            if c["near_miss_share_of_misses"] is not None
        ]
        print(
            f"  median actual recall: {fmt_pct(statistics.median(actuals))} | "
            f"median ceiling recall: {fmt_pct(statistics.median(ceils))} | "
            f"median near-miss share of misses: {fmt_pct(statistics.median(shares)) if shares else 'n/a'}"
        )
        dump[f"{label.lower().replace(' ', '_')}_recall_ceiling"] = ceilings

    return dump


def fact_detail_rows_for(
    truth: Truth, stats: SourceStats, runs_data: list[dict[str, Any]], fact_ids: list[str]
) -> list[dict[str, Any]]:
    return [
        build_fact_detail(truth, stats, runs_data, fid)
        for fid in sorted(fact_ids, key=lambda f: (len(f), f))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--json", type=Path, default=None, help="also dump full data as JSON here")
    args = parser.parse_args()

    truth = load_truth()
    stats = SourceStats(truth, DEFAULT_CORPUS_ROOT)

    kimi_runs = load_model_runs(args.bench_root, "kimi")
    nova_runs = load_model_runs(args.bench_root, "nova")

    dump = render_report(truth, stats, kimi_runs, nova_runs)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
        print(f"\n[wrote full data dump to {args.json}]")


if __name__ == "__main__":
    main()
