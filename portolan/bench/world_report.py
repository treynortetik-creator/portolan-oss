"""Markdown and JSON for a v2 world bench run and for a dispute run. Every
number printed here is read off a `Tally` or a dispute summary; nothing is
computed in prose."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from portolan.bench.world_score import BREAKDOWNS, Tally, WorldScore

__all__ = ["render_disputes_markdown", "render_world_markdown", "write_world_report"]

_LABEL_SIDE = {"label_source", "directive", "reference_mode"}
_TITLES = {
    "fact_type": "By fact type",
    "doc_kind": "By document kind",
    "label_source": "By label source (label side)",
    "directive": "By directive (label side)",
    "reference_mode": "By reference mode (label side)",
    "noise": "By noise operation",
}


def _pct(n: int | float, d: int | float) -> str:
    if not d:
        return "n/a"
    return f"{100.0 * n / d:.1f}% ({int(n)}/{int(d)})"


def _row(name: str, t: Tally) -> str:
    return (
        f"| {name} | {t.produced} "
        f"| {_pct(t.quote_exact + t.quote_normalized, t.produced)} "
        f"| {_pct(t.quote_exact, t.produced)} "
        f"| {_pct(t.substantiated_count, t.produced)} "
        f"| {_pct(t.omitted, t.labeled)} "
        f"| {t.decoy_hits - t.decoy_overturned} "
        f"| {t.type_confusions} "
        f"| {t.off_span - t.off_span_overturned} "
        f"| {t.decoy_overturned + t.off_span_overturned} |"
    )


_HEADER = (
    "| Bucket | Produced | Quote verbatim | Quote exact | Substantiated | Omission "
    "| Decoy hits | Type confusions | Off-span | Overturned by judges |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)

_LABEL_HEADER = (
    "| Bucket | Labeled | Found | Omission | Omitted with a type confusion "
    "| Matched produced facts | Quote verbatim of facts that hit these labels |\n"
    "|---|---|---|---|---|---|---|"
)


def _label_row(name: str, t: Tally) -> str:
    return (
        f"| {name} | {t.labeled} | {t.labeled_found} | {_pct(t.omitted, t.labeled)} "
        f"| {t.labeled_confused} | {t.matched} "
        f"| {_pct(t.quote_exact + t.quote_normalized, t.produced)} |"
    )


def render_world_markdown(score: WorldScore) -> str:
    o = score.overall
    lines = [
        f"# World extraction bench: {score.world_id}, {score.date.isoformat()}",
        "",
        f"- World directory: `{score.world_dir}` (read from `{score.input_format}`)",
        f"- Backend: `{score.backend}`, model `{score.extractor_model}`, prompt "
        f"`{score.prompt_variant}`, samples {score.samples}",
        f"- Sources scored: {score.sources_scored}; failed: {len(score.sources_failed)}; "
        f"skipped by the reader: {len(score.sources_skipped)}",
        f"- Label rows used: {score.labels_used} ({o.labeled} labeled facts); excluded: "
        f"{json.dumps(score.labels_excluded, sort_keys=True)}",
        f"- Decoys used: {score.decoys_used}; excluded: "
        f"{json.dumps(score.decoys_excluded, sort_keys=True)}",
        f"- Extraction tokens in/out: {score.tokens_input}/{score.tokens_output}; "
        f"cost ${score.cost_usd:.4f}; wall {score.wall_time_s:.1f} s",
        "",
        "## The three numbers",
        "",
        f"1. **Quote-verbatim**: {_pct(o.quote_exact + o.quote_normalized, o.produced)} "
        f"(exact {_pct(o.quote_exact, o.produced)}, normalized "
        f"{_pct(o.quote_normalized, o.produced)}, stitched {o.quote_stitched}, "
        f"unresolved {o.quote_unresolved}, no quote {o.quote_missing})",
        f"2. **Substantiated**: {_pct(o.substantiated_count, o.produced)} "
        f"(matched a label {o.matched}, of them duplicates {o.duplicates}; "
        f"judges ruled real {o.judge_supported})",
        f"3. **Omission**: {_pct(o.omitted, o.labeled)}. **Invention**: "
        f"{_pct(o.invented_count, o.produced)} (decoy hits "
        f"{o.decoy_hits - o.decoy_overturned}, off-span {o.off_span - o.off_span_overturned}); "
        f"type confusions {o.type_confusions}, shown apart",
        "",
    ]
    for dim in BREAKDOWNS:
        buckets = score.breakdowns.get(dim, {})
        lines += [f"## {_TITLES[dim]}", ""]
        if dim in _LABEL_SIDE:
            lines.append(_LABEL_HEADER)
            lines += [_label_row(k, t) for k, t in buckets.items() if t.labeled]
        else:
            lines.append(_HEADER)
            lines += [_row(k, t) for k, t in buckets.items()]
        lines.append("")
    lines += ["## Type confusions (label type -> produced type)", ""]
    if score.type_confusions:
        lines += ["| Pair | Count |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in score.type_confusions.items()]
    else:
        lines.append("None.")
    lines.append("")
    if score.disputes is not None:
        lines += _dispute_lines(score.disputes.model_dump())
    if score.sources_failed:
        lines += ["## Failed sources", ""]
        lines += [f"- `{k}`: {v}" for k, v in sorted(score.sources_failed.items())]
        lines.append("")
    return "\n".join(lines) + "\n"


def _dispute_lines(d: dict[str, Any]) -> list[str]:
    lines = [
        "## Three-judge disputes",
        "",
        f"- Judges: {', '.join(d.get('judge_models') or []) or 'none'}",
        f"- Estimate ${d.get('estimate_usd', 0.0):.4f} expected, "
        f"${d.get('ceiling_usd', 0.0):.4f} ceiling; spent "
        f"${d.get('spent_usd', 0.0):.4f} in {d.get('calls', 0)} calls; cap "
        f"${d.get('max_usd', 0.0):.2f}; stopped: {d.get('stopped') or 'no'}",
        f"- Proposed corrections: `{d.get('corrections_path')}` (never applied)",
        "",
        "| Category | Disputed | Judged | Overturned (ruled real) | Upheld | Not judged |",
        "|---|---|---|---|---|---|",
    ]
    for cat, row in (d.get("by_category") or {}).items():
        lines.append(
            f"| {cat} | {row['disputed']} | {row['judged']} | {row['overturned']} "
            f"| {row['upheld']} | {row['not_judged']} |"
        )
    lines.append("")
    return lines


def render_disputes_markdown(
    title: str, summary: dict[str, Any], outcomes: list[dict[str, Any]]
) -> str:
    lines = [f"# {title}", "", *_dispute_lines(summary), "## Every dispute", ""]
    lines += [
        "| Source | Fact | Category | Type | Claim | Votes | Decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        d = o["dispute"]
        votes = ", ".join(f"{v['judge_model']}={v['effective']}" for v in o["votes"]) or "-"
        claim = d["claim"].replace("|", "/")[:120]
        lines.append(
            f"| {d['source_id']} | {d['produced_id']} | {d['category']} | {d['fact_type']} "
            f"| {claim} | {votes} | {o['decision']} |"
        )
    return "\n".join(lines) + "\n"


def write_world_report(score: WorldScore, out_dir: Path | str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = f"world-bench-{score.date.isoformat()}-{score.backend}"
    md = out / f"{base}.md"
    js = out / f"{base}.json"
    md.write_text(render_world_markdown(score), encoding="utf-8")
    payload = json.loads(score.model_dump_json())
    payload["overall_rates"] = score.overall.rates()
    payload["breakdown_rates"] = {
        dim: {k: t.rates() for k, t in buckets.items()} for dim, buckets in score.breakdowns.items()
    }
    js.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return md, js
