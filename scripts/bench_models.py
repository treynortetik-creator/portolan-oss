"""Run the extraction bench on many models at once and aggregate the results.

Usage:
    set -a; . ./.env; set +a
    uv run python scripts/bench_models.py --out var/bench/compare --concurrency 4 \
        amazon/nova-micro-v1 amazon/nova-lite-v1 openai/gpt-5.6-luna ...

Each model runs as its own subprocess of `python -m portolan.bench extract --backend openrouter`
in parallel; when all finish, the JSON reports are aggregated into one markdown table written to
`<out>/model-comparison-<date>.md`. Absolute dates; every number comes from the JSON, never retyped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


def slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def launch(
    model: str, out: Path, concurrency: int, corpus: str, extra: list[str]
) -> subprocess.Popen:
    d = out / slug(model)
    d.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "portolan.bench",
        "extract",
        "--backend",
        "openrouter",
        "--model",
        model,
        "--corpus",
        corpus,
        "--out",
        str(d) + "/",
        "--concurrency",
        str(concurrency),
        *extra,
    ]
    # The child gets its own copy of the descriptor, so the parent's can close on return.
    with open(d / "run.log", "w") as log:
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def read_report(d: Path) -> dict | None:
    files = sorted(d.glob("bench-*.json"))
    if not files:
        return None
    j = json.loads(files[-1].read_text())
    ex = j.get("extraction", {})

    def g(*keys):
        for k in keys:
            if k in ex and ex[k] is not None:
                return ex[k]
        return None

    unplanted = ex.get("unplanted") if isinstance(ex.get("unplanted"), list) else []
    non_leak = [u for u in unplanted if not u.get("is_distractor_leak")]
    plausible = sum(1 for u in non_leak if u.get("in_source_text"))
    invented = sum(1 for u in non_leak if not u.get("in_source_text"))
    produced = g("produced_total")
    # Fix instructions item 3: invented divided by produced -- same
    # metric portolan.bench.report now puts in a single run's own headline,
    # here across every model in the comparison. None (rendered "?") when
    # nothing was produced, same convention as every other field here.
    invented_rate = (invented / produced) if produced else None
    return {
        "recall": g("recall"),
        "precision": g("precision"),
        "planted": g("planted_total"),
        "matched": g("matched_count"),
        "produced": produced,
        "invented": invented,
        "invented_rate": invented_rate,
        "plausible": plausible,
        "leakage": g("distractor_leakage"),
        "failed": len(ex.get("failed_sources") or {}),
        "in_tokens": g("tokens_input"),
        "out_tokens": g("tokens_output"),
        "cost": g("cost_estimate"),
        "wall_s": g("wall_time_s"),
    }


def fmt(v, kind: str = "") -> str:
    if v is None:
        return "?"
    if kind == "pct":
        return f"{100 * float(v):.1f}%"
    if kind == "usd":
        return f"${float(v):.3f}"
    if kind == "s":
        return f"{float(v):.0f}s"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default="var/bench/compare")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--corpus", default="corpus/out")
    ap.add_argument("--extra", default="", help="extra args passed to every bench run")
    args = ap.parse_args()
    out = Path(args.out)
    extra = args.extra.split() if args.extra else []
    procs = {m: launch(m, out, args.concurrency, args.corpus, extra) for m in args.models}
    print(f"launched {len(procs)} model runs", flush=True)
    for m, p in procs.items():
        p.wait()
        print(f"done {m} exit {p.returncode}", flush=True)
    rows = []
    for m in args.models:
        r = read_report(out / slug(m))
        rows.append((m, r))
    date = dt.date.today().isoformat()
    lines = [
        f"# Model comparison — extraction bench — {date}",
        "",
        f"Corpus `{args.corpus}` (38 planted facts, 4 distractors), OpenRouter, {args.concurrency} sources in parallel per model, all models run at once. Numbers copied from each run's JSON report; nothing retyped.",
        "",
        "| model | recall | precision | produced | invented | invented rate | plausible unplanted | leakage | failed sources | in tokens | out tokens | cost | wall |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, r in rows:
        if r is None:
            lines.append(f"| {m} | run failed, see run.log | | | | | | | | | | | |")
            continue
        lines.append(
            f"| {m} | {fmt(r['recall'], 'pct')} | {fmt(r['precision'], 'pct')} | {fmt(r['produced'])} | "
            f"{fmt(r['invented'])} | {fmt(r['invented_rate'], 'pct')} | {fmt(r['plausible'])} | "
            f"{fmt(r['leakage'])} | {fmt(r['failed'])} | "
            f"{fmt(r['in_tokens'])} | {fmt(r['out_tokens'])} | {fmt(r['cost'], 'usd')} | {fmt(r['wall_s'], 's')} |"
        )
    md = out / f"model-comparison-{date}.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")
    missing = [m for m, r in rows if r is None]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
