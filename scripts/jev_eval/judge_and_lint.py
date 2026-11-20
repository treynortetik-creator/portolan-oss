#!/usr/bin/env python3
"""Track A eval: is Jev (TypeSafe AI) a viable theme-resolution judge and
lint gate for Portolan? (var/jev-typesafe-brief-2026-09-16.md secs 3-6)

Compares Jev against the current judge path -- Nova Lite via OpenRouter,
reusing `portolan.report.themes`'s real `RESOLUTION_JUDGE_SYSTEM` prompt and
`_judge_resolution` function verbatim, not a re-implementation -- on the same
40 labeled (theme, candidate claim) pairs (`scripts/jev_eval/data.py`,
`RESOLUTION_PAIRS`). Also runs 60 labeled lint claims through Jev alone on
three noul gates (attribution / factual / about-the-named-account).

Run (from the repo root, inside the `jev-judge` worktree):

    set -a; source .env; set +a
    uv run python scripts/jev_eval/judge_and_lint.py

Every raw Jev/Nova response is cached under var/jev-eval/ (gitignored) keyed
by a hash of the exact request, so re-running this script after a code fix
costs nothing further for requests it already has an answer for. Final
metrics are written to var/jev-eval/results.json and printed to stdout --
every number in the eval report is read from that file, never eyeballed.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from portolan.llm.openrouter import OpenRouterClient  # noqa: E402
from portolan.metering import InMemorySink  # noqa: E402
from portolan.report import themes  # noqa: E402
from portolan.training.spend import SpendMeter, default_spend_path  # noqa: E402
from scripts.jev_eval.data import LINT_CLAIMS, RESOLUTION_PAIRS  # noqa: E402

CACHE_DIR = REPO_ROOT / "var" / "jev-eval"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_PRICE_PER_MILLION_INPUT = 0.042  # var/jev-typesafe-brief-2026-09-16.md sec 4

NOVA_MODEL = "amazon/nova-lite-v1"  # OpenRouter id, portolan/pricing.py row


def _hash(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _cache_get(prefix: str, key: str) -> dict | None:
    path = CACHE_DIR / f"{prefix}_{key}.json"
    if path.exists():
        data = json.loads(path.read_text())
        data["_cached"] = True
        return data
    return None


def _cache_put(prefix: str, key: str, data: dict) -> None:
    path = CACHE_DIR / f"{prefix}_{key}.json"
    path.write_text(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Jev HTTP call
# ---------------------------------------------------------------------------


def call_jev(
    client: httpx.Client, api_key: str, *, state: str, questions: dict, cache_key: str
) -> dict:
    cached = _cache_get("jev", cache_key)
    if cached is not None:
        return cached

    body = {"state": state, "model": JEV_MODEL, "questions": questions}
    last_exc: Exception | None = None
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            resp = client.post(
                JEV_URL,
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60.0,
            )
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Jev call failed after retries: {exc}") from exc
        latency = time.perf_counter() - t0
        if resp.status_code == 200:
            data = resp.json()
            data["_latency_s"] = latency
            data["_cached"] = False
            _cache_put("jev", cache_key, data)
            return data
        if resp.status_code in (429, 529) and attempt < 2:
            time.sleep(2 * (attempt + 1))
            continue
        raise RuntimeError(f"Jev HTTP {resp.status_code}: {resp.text[:500]}")
    raise RuntimeError(f"Jev call failed after retries: {last_exc}")


# ---------------------------------------------------------------------------
# Track A1: theme-resolution judge
# ---------------------------------------------------------------------------


def _member_state(theme) -> tuple[str, list[str]]:
    accounts = [a for a, _ in theme.members]
    texts = [t for _, t in theme.members]
    return theme.name, [f'- ({a}) "{t}"' for a, t in zip(accounts, texts, strict=True)]


def build_jev_resolution_request(theme, pair) -> tuple[str, dict]:
    theme_name, member_lines = _member_state(theme)
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(pair.sentences))
    state = (
        f'Theme: "{theme_name}"\n\n'
        f"Member claims already attached to this theme:\n" + "\n".join(member_lines) + "\n\n"
        f"Candidate claim, from {pair.account or 'an unnamed internal source'}, "
        f"dated {pair.claim_date:%Y-%m-%d}:\n{numbered}"
    )
    questions = {
        "resolves": {
            "type": "noul",
            "instructions": (
                "Look at the theme and its member claims, then the candidate claim below them. "
                "Does the candidate claim state that THIS theme's problem is now resolved or over, "
                "for the account it is about -- not a different problem, and not a generic positive "
                "remark that merely shares a word with the theme?"
            ),
        },
        "evidence_sentence": {
            "type": "choice",
            "instructions": (
                "The candidate claim's sentences are numbered. Which single numbered sentence best "
                "supports your answer to whether the theme's problem is resolved? Pick the most "
                "directly relevant sentence even if you answered no."
            ),
            "criteria": {str(i + 1): s for i, s in enumerate(pair.sentences)},
        },
        "directness": {
            "type": "score",
            "instructions": (
                "How directly does the candidate claim address whether this specific theme's problem "
                "is resolved for this account?"
            ),
            "criteria": [
                "Unrelated to this theme or this account",
                "On-topic for this theme but does not say whether it is resolved",
                "Directly states whether this theme's problem is resolved for this account",
            ],
        },
    }
    return state, questions


def call_nova_resolution(llm: OpenRouterClient, theme, pair, *, cache_key: str) -> dict:
    cached = _cache_get("nova", cache_key)
    if cached is not None:
        return cached

    state = themes._ThemeState(
        slug=theme.slug,
        dominant_fact_type=pair.fact_type,
        product_families=set(),
        product_slugs=set(),
        accounts={a for a, _ in theme.members},
        member_claim_ids=set(range(len(theme.members))),
        member_texts=[t for _, t in theme.members],
        member_accounts=[a for a, _ in theme.members],
    )
    candidate_text = " ".join(pair.sentences)
    candidate = themes._ClaimCandidate(
        id=hash(pair.pair_id) % 100000,
        note_slug=f"signal-eval-{pair.pair_id.lower()}",
        fact_type=pair.fact_type,
        claim_text=candidate_text,
        source_slug=f"eval-source-{pair.pair_id.lower()}",
        locator="p1",
        quote=None,
        recorded=pair.claim_date,
        valid_from=None,
        account_slug=pair.account,
    )
    t0 = time.perf_counter()
    verdict, resolved_text = themes._judge_resolution(
        llm, theme_name=theme.name, state=state, candidate=candidate, model=NOVA_MODEL
    )
    latency = time.perf_counter() - t0
    event = llm.sink.events[-1]
    result = {
        "resolves": verdict.resolves,
        "evidence": verdict.evidence,
        "reason": verdict.reason,
        "candidate_text": resolved_text,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "cached_tokens": event.cached_tokens,
        "cost_estimate": event.cost_estimate,
        "_latency_s": latency,
        "_cached": False,
    }
    _cache_put("nova", cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Track A2: lint gates
# ---------------------------------------------------------------------------


def build_jev_lint_request(claim) -> tuple[str, dict]:
    state = f'Account under review: "{claim.presented_account}"\n\nClaim: "{claim.text}"'
    questions = {
        "attributed": {
            "type": "noul",
            "instructions": (
                "Does the claim explicitly attribute what it reports to a named person or source "
                '(e.g. "X said...", "per Y", a named speaker or a named feed), rather than stating '
                "it as a bare, unsourced assertion?"
            ),
        },
        "factual": {
            "type": "noul",
            "instructions": (
                "Is this a factual claim -- something that concretely happened or was concretely "
                "stated -- rather than an opinion, guess, or speculation about the future?"
            ),
        },
        "about_account": {
            "type": "noul",
            "instructions": (
                'Is this claim specifically about the account named as "Account under review" '
                "above, rather than about a different account or no account in particular?"
            ),
        },
    }
    return state, questions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _confusion(preds: list[bool], truth: list[bool]) -> dict:
    tp = sum(p and t for p, t in zip(preds, truth, strict=True))
    tn = sum(not p and not t for p, t in zip(preds, truth, strict=True))
    fp = sum(p and not t for p, t in zip(preds, truth, strict=True))
    fn = sum(not p and t for p, t in zip(preds, truth, strict=True))
    n = len(preds)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def _latency_stats(latencies: list[float]) -> dict:
    if not latencies:
        return {"median_s": None, "p95_s": None, "n": 0}
    ordered = sorted(latencies)
    median = statistics.median(ordered)
    idx = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    p95 = ordered[idx]
    return {"median_s": median, "p95_s": p95, "n": len(ordered)}


def _bucket(noul: float) -> str:
    if noul >= 0.9:
        return "0.9-1.0"
    if noul >= 0.7:
        return "0.7-0.9"
    if noul >= 0.5:
        return "0.5-0.7"
    return "<0.5"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    typesafe_key = os.environ.get("TYPESAFE_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not typesafe_key:
        raise SystemExit("TYPESAFE_API_KEY not set (source .env first)")
    if not openrouter_key:
        raise SystemExit("OPENROUTER_API_KEY not set (source .env first)")

    jev_client = httpx.Client()
    sink = InMemorySink()
    nova_llm = OpenRouterClient(
        spend_key="jev_eval",
        spend_meter=SpendMeter(default_spend_path(REPO_ROOT / "var")),
        api_key=openrouter_key,
        sink=sink,
        tenant="jev-eval",
    )

    # ---------------- Track A1: theme-resolution judge ----------------
    row_records: list[dict] = []
    for pair in RESOLUTION_PAIRS:
        state, questions = build_jev_resolution_request(pair.theme, pair)
        jev_key = _hash(pair.pair_id + "|" + state + "|" + json.dumps(questions, sort_keys=True))
        jev_resp = call_jev(
            jev_client, typesafe_key, state=state, questions=questions, cache_key=jev_key
        )

        nova_key = _hash(pair.pair_id + "|nova|" + pair.theme.slug)
        nova_resp = call_nova_resolution(nova_llm, pair.theme, pair, cache_key=nova_key)

        jev_noul = jev_resp["answers"]["resolves"]["noul"]
        jev_choice = jev_resp["answers"]["evidence_sentence"]["choice"]
        jev_score = jev_resp["answers"]["directness"]["score"]
        jev_predicted = jev_noul >= 0.5

        candidate_text = " ".join(pair.sentences)
        nova_predicted = bool(
            nova_resp["resolves"]
            and nova_resp["evidence"]
            and nova_resp["evidence"] in candidate_text
        )

        row_records.append(
            {
                "pair_id": pair.pair_id,
                "fact_id": pair.fact_id,
                "theme": pair.theme.name,
                "account": pair.account,
                "category": pair.category,
                "label": pair.label,
                "resolving_sentence": pair.resolving_sentence,
                "jev_noul": jev_noul,
                "jev_choice": jev_choice,
                "jev_score": jev_score,
                "jev_predicted": jev_predicted,
                "jev_evidence_matches_ground_truth": (
                    pair.resolving_sentence is not None
                    and str(pair.resolving_sentence) == jev_choice
                )
                if pair.resolving_sentence is not None
                else None,
                "jev_input_tokens": jev_resp["usage"]["input_tokens"],
                "jev_output_tokens": jev_resp["usage"]["output_tokens"],
                "jev_latency_s": jev_resp.get("_latency_s"),
                "jev_cached": jev_resp.get("_cached", False),
                "nova_resolves_raw": nova_resp["resolves"],
                "nova_evidence": nova_resp["evidence"],
                "nova_predicted": nova_predicted,
                "nova_input_tokens": nova_resp["input_tokens"],
                "nova_output_tokens": nova_resp["output_tokens"],
                "nova_cost_estimate": nova_resp["cost_estimate"],
                "nova_latency_s": nova_resp.get("_latency_s"),
                "nova_cached": nova_resp.get("_cached", False),
            }
        )

    jev_preds = [r["jev_predicted"] for r in row_records]
    nova_preds = [r["nova_predicted"] for r in row_records]
    truth = [r["label"] for r in row_records]

    jev_conf = _confusion(jev_preds, truth)
    nova_conf = _confusion(nova_preds, truth)

    jev_latencies = [
        r["jev_latency_s"] for r in row_records if not r["jev_cached"] and r["jev_latency_s"]
    ]
    nova_latencies = [
        r["nova_latency_s"] for r in row_records if not r["nova_cached"] and r["nova_latency_s"]
    ]

    jev_cost = (
        sum(r["jev_input_tokens"] for r in row_records) * JEV_PRICE_PER_MILLION_INPUT / 1_000_000
    )
    nova_cost = sum(r["nova_cost_estimate"] or 0.0 for r in row_records)

    buckets: dict[str, list[bool]] = {"<0.5": [], "0.5-0.7": [], "0.7-0.9": [], "0.9-1.0": []}
    for r in row_records:
        b = _bucket(r["jev_noul"])
        buckets[b].append(r["jev_predicted"] == r["label"])
    bucket_table = {
        b: {"n": len(v), "accuracy": (sum(v) / len(v) if v else None)} for b, v in buckets.items()
    }

    positives = [r for r in row_records if r["label"]]
    evidence_hits = [r["jev_evidence_matches_ground_truth"] for r in positives]
    evidence_validity_rate = (
        sum(1 for h in evidence_hits if h) / len(evidence_hits) if evidence_hits else None
    )

    # ---------------- Track A2: lint gates ----------------
    lint_records: list[dict] = []
    for claim in LINT_CLAIMS:
        state, questions = build_jev_lint_request(claim)
        key = _hash(claim.claim_id + "|" + state)
        resp = call_jev(jev_client, typesafe_key, state=state, questions=questions, cache_key=key)
        lint_records.append(
            {
                "claim_id": claim.claim_id,
                "fact_id": claim.fact_id,
                "hand_authored": claim.hand_authored,
                "attributed_true": claim.attributed,
                "factual_true": claim.factual,
                "about_account_true": claim.about_account,
                "attributed_noul": resp["answers"]["attributed"]["noul"],
                "factual_noul": resp["answers"]["factual"]["noul"],
                "about_account_noul": resp["answers"]["about_account"]["noul"],
                "input_tokens": resp["usage"]["input_tokens"],
                "output_tokens": resp["usage"]["output_tokens"],
                "latency_s": resp.get("_latency_s"),
                "cached": resp.get("_cached", False),
            }
        )

    def _gate_conf(field_pred: str, field_truth: str) -> dict:
        preds = [r[field_pred] >= 0.5 for r in lint_records]
        truths = [r[field_truth] for r in lint_records]
        return _confusion(preds, truths)

    lint_conf = {
        "attributed": _gate_conf("attributed_noul", "attributed_true"),
        "factual": _gate_conf("factual_noul", "factual_true"),
        "about_account": _gate_conf("about_account_noul", "about_account_true"),
    }
    lint_latencies = [r["latency_s"] for r in lint_records if not r["cached"] and r["latency_s"]]
    lint_cost = (
        sum(r["input_tokens"] for r in lint_records) * JEV_PRICE_PER_MILLION_INPUT / 1_000_000
    )
    hand_authored_count = sum(1 for c in LINT_CLAIMS if c.hand_authored)

    results = {
        "track_a1_theme_judge": {
            "n_pairs": len(row_records),
            "n_positive": sum(truth),
            "n_negative": len(truth) - sum(truth),
            "jev": {
                "confusion": jev_conf,
                "false_positives": jev_conf["fp"],
                "latency": _latency_stats(jev_latencies),
                "total_cost_usd": jev_cost,
                "confidence_buckets": bucket_table,
                "evidence_validity_rate": evidence_validity_rate,
                "evidence_validity_n": len(evidence_hits),
            },
            "nova_lite": {
                "confusion": nova_conf,
                "false_positives": nova_conf["fp"],
                "latency": _latency_stats(nova_latencies),
                "total_cost_usd": nova_cost,
            },
            "rows": row_records,
        },
        "track_a2_lint_gates": {
            "n_claims": len(lint_records),
            "n_hand_authored": hand_authored_count,
            "gates": lint_conf,
            "latency": _latency_stats(lint_latencies),
            "total_cost_usd": lint_cost,
            "rows": lint_records,
        },
        "total_cost_usd": jev_cost + nova_cost + lint_cost,
    }

    out_path = CACHE_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    print(
        json.dumps(
            {
                "track_a1_jev": {
                    k: v for k, v in results["track_a1_theme_judge"]["jev"].items() if k != "rows"
                },
                "track_a1_nova": {
                    k: v
                    for k, v in results["track_a1_theme_judge"]["nova_lite"].items()
                    if k != "rows"
                },
                "track_a2_gates": results["track_a2_lint_gates"]["gates"],
                "track_a2_latency": results["track_a2_lint_gates"]["latency"],
                "track_a2_cost_usd": results["track_a2_lint_gates"]["total_cost_usd"],
                "total_cost_usd": results["total_cost_usd"],
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
