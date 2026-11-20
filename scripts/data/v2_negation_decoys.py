"""Regenerate every v2 pack's negation decoy class under the 2026-09-23
decision (`01-spec/decisions-log.md`): a negated competitor statement is a
`competitor_mention` fact, so a negation decoy must be a negation that
carries no durable fact.

What changes per pack, and who writes it:

| Field | Change | Written by |
|---|---|---|
| `near_type` | `request` (a negated request or offer: nothing is asked, promised or needed) | structure |
| `forbidden_patterns` | `[competitor_evaluation]`, enforced by WP4 check 5 | structure |
| `description`, `example` | regenerated | the open-weight model (`MODEL`), curated by code: kept verbatim or dropped, never edited |
| `text_source` | model, date, sha256 of every raw response | code |

The brief below is an instruction to the model, never text that reaches a
document; the writer prompt shows only the curated `description`
(`writer_prompt._script_lines`). Spend goes on the shared meter under
`v2_packs` (the packs' own line) and is checked before every call.

    uv run python scripts/data/v2_negation_decoys.py --env-root <checkout with .env>
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from corpus.generator.v2.packs.schema import load_pack  # noqa: E402
from corpus.generator.v2.verify import forbidden_pattern_hits  # noqa: E402
from portolan.training.spend import SpendMeter, default_spend_path, shared_var_root  # noqa: E402
from scripts.data.v2_packs import claim_templates as ct  # noqa: E402

PACKS_DIR = REPO_ROOT / "corpus" / "generator" / "v2" / "packs"
MODEL = "qwen/qwen3-32b"  # the model every pack's claim templates came from
CORPUS_KEY = "v2_packs"
BUDGET_USD = 0.10  # this run's growth of the line
NEAR_TYPE = "request"
FORBIDDEN = ["competitor_evaluation"]
MAX_ATTEMPTS = 6
MAX_WORDS = 35
# qwen3-32b can spend a small budget on reasoning and reply with nothing.
MAX_TOKENS = 2000

# Words that make a "negation" a status report (praise, a complaint or a
# risk) or a choice (a decision) rather than a statement of no fact.
_STATUS_OR_CHOICE = re.compile(
    r"\b(?:issues?|problems?|alarms?|errors?|outages?|downtime|bugs?|glitch\w*|broken|"
    r"working|works|fine|smooth\w*|stable|delays?|late|cancel\w*|declin\w*|renew\w*|"
    r"price\w*|pricing|discount\w*|budget\w*|contract\w*|decid\w*|decision\w*)\b",
    re.IGNORECASE,
)

BRIEF = """Vertical: {vertical}
What the vendor sells: {vendor_what}
{exclusion}
A synthetic-data generator plants near-miss statements in business documents:
sentences that sound like a fact about the customer but state none. Describe
and illustrate ONE such pattern:

THE PATTERN: a negation that states no fact. Someone says they are NOT asking
for anything, are NOT promising anything yet, or that nothing is needed from
the other side right now -- a polite "no need" that records no request, no
commitment, no decision and no status.

The sentence must NOT:
- mention a competitor, another vendor, supplier, platform or product, or
  looking at, evaluating, switching to, sticking with or ruling out anything;
- say how anything is performing (no "no issues", "no problems", "working fine");
- decline, cancel or renew anything, or mention price, budget or contracts;
- contain a number, a date, or a person's or company's name.

Earlier attempts that were rejected, and why:
{rejected}

Respond with ONLY this JSON:
{{"description": "<one plain sentence describing the pattern for a writer>",
  "example": "<one sentence a real {vertical} customer might say, informal, first person>"}}
"""
SYSTEM = (
    "You write short, plain, business-register text for a B2B healthcare-software "
    "synthetic data generator. You output ONLY valid JSON."
)


def reject_reason(text: str, *, is_senior_living: bool, example: bool) -> str | None:
    """Why a candidate description or example is not kept (curation never
    edits a model sentence; it keeps it verbatim or drops it). A description
    may name what the pattern is not ("no request, no decision"); an example
    may not say any of it."""
    if not text.strip():
        return "empty"
    if ct._PLACEHOLDER_RE.search(text):
        return "curly-brace placeholder"
    if len(text.split()) > MAX_WORDS:
        return f"over {MAX_WORDS} words"
    if re.search(r"\d", text):
        return "a number"
    hits = forbidden_pattern_hits(text, FORBIDDEN)
    if hits:
        return f"forbidden pattern {hits[0]}"
    status = _STATUS_OR_CHOICE.search(text) if example else None
    if status:
        return f"status or choice word {status.group(0)!r}"
    if is_senior_living and ct.FORBIDDEN_SENIOR_LIVING_PATTERN.search(text):
        return "senior-living excluded vocabulary"
    return None


def negation_class_id(pack_path: Path) -> str:
    pack = load_pack(pack_path)
    ids = [c.id for c in pack.decoy_classes if c.id.split("_")[0] == "negation"]
    if len(ids) != 1:
        raise ValueError(f"{pack_path.name}: expected one negation decoy class, found {ids}")
    return ids[0]


def generate(
    client: Any, spend: SpendMeter, baseline: float, pack_path: Path
) -> tuple[dict[str, str], list[str]]:
    from scripts.data import v2_gen_pack_templates as gen

    pack = load_pack(pack_path)
    is_sl = pack.id == "senior_living_tech"
    exclusion = f"\nHARD EXCLUSION: {gen.SENIOR_LIVING_EXCLUSION}\n" if is_sl else ""
    rejected: list[str] = []
    digests: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        if spend.corpus_usd(CORPUS_KEY) - baseline >= BUDGET_USD:
            raise RuntimeError(f"{CORPUS_KEY} grew by ${BUDGET_USD:.2f} in this run; stopping")
        user = BRIEF.format(
            vertical=pack.name,
            vendor_what=pack.vendor_what,
            exclusion=exclusion,
            rejected="\n".join(f"- {r}" for r in rejected) or "(none)",
        )
        result, digest = gen._call(
            client,
            spend,
            pack.id,
            "negation-decoy-2026-09-23",
            SYSTEM,
            user,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=max(0.3, 0.9 - 0.15 * attempt),
        )
        digests.append(digest)
        try:
            data = ct.extract_json(result.text or "")
        except ValueError:
            if result.text:
                rejected.append("(reply was not JSON)")
            continue
        fields = {k: str(data.get(k) or "").strip() for k in ("description", "example")}
        reasons = {
            k: reject_reason(v, is_senior_living=is_sl, example=k == "example")
            for k, v in fields.items()
        }
        bad = {k: r for k, r in reasons.items() if r}
        if not bad:
            return fields, digests
        rejected += [f"{k} {fields[k]!r}: {r}" for k, r in bad.items()]
        print(f"[{pack.id}] attempt {attempt + 1} rejected: {bad}", flush=True)
    raise RuntimeError(f"{pack.id}: no candidate passed curation in {MAX_ATTEMPTS} attempts")


def patch_pack(
    pack_path: Path, class_id: str, fields: dict[str, str], digests: list[str], generated: str
) -> None:
    """Rewrite the class's `near_type`, `description` and `example` lines and
    add its `forbidden_patterns` and `text_source`, touching no other line."""
    from scripts.data.v2_packs import render

    lines = pack_path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == f"- id: {class_id}")
    end = start + 1
    while end < len(lines) and lines[end].startswith("    ") and not lines[end].startswith("  - "):
        end += 1
    indent = "    "
    block = [
        lines[start],
        f"{indent}near_type: {NEAR_TYPE}",
        f"{indent}description: {render._yq(fields['description'])}",
        f"{indent}example: {render._yq(fields['example'])}",
        f"{indent}forbidden_patterns: [{', '.join(FORBIDDEN)}]",
        f"{indent}text_source:",
        f"{indent}  model: {render._yq(MODEL)}",
        f"{indent}  generated: {render._yq(generated)}",
        f"{indent}  raw_response_sha256:",
        *[f"{indent}    - {render._yq(d)}" for d in digests],
    ]
    pack_path.write_text("\n".join([*lines[:start], *block, *lines[end:]]) + "\n", encoding="utf-8")
    load_pack(pack_path)  # still validates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--pack", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true", help="call and curate, write nothing")
    args = parser.parse_args(argv)

    from corpus.generator.prose import build_openrouter_client
    from scripts.data import v2_gen_pack_templates as gen

    # Raw responses go to the machine's shared `var/`, next to the packs' earlier ones.
    gen.RAW_DIR = shared_var_root(REPO_ROOT / "var") / "data" / "v2_packs" / "raw"
    spend = SpendMeter(path=default_spend_path(REPO_ROOT / "var"))
    baseline = spend.corpus_usd(CORPUS_KEY)
    client = build_openrouter_client(args.env_root)
    client.sink = gen.SpendGatedSink(spend)
    generated = date.today().isoformat()
    paths = sorted(PACKS_DIR.glob("*.yaml"))
    if args.pack:
        paths = [p for p in paths if p.stem in args.pack]
    for path in paths:
        class_id = negation_class_id(path)
        fields, digests = generate(client, spend, baseline, path)
        print(f"[{path.stem}] {class_id}: {fields}", flush=True)
        if not args.dry_run:
            patch_pack(path, class_id, fields, digests, generated)
    print(f"{CORPUS_KEY} spend this run: ${spend.corpus_usd(CORPUS_KEY) - baseline:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
