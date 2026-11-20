"""``python -m corpus.generator.cli generate --out corpus/out --mode auto|api|manual``

Orchestrates the whole pipeline: load the truth table, build briefs, produce
every source (structured + prose), run the marker pass, write the corpus and
the manifest, validate, and report counts. This is the only module allowed
to print — everything else stays quiet so it can be imported by tests.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from corpus.generator import manifest as manifest_mod
from corpus.generator import markers, prose, structured, worlds
from corpus.generator.briefs import (
    MULTI_MENTION_FACTS,
    Brief,
    Registry,
    build_briefs,
    build_calls,
    build_chat_briefs,
    build_crm,
    build_crm_notes,
    build_emails,
    build_tickets,
    load_truth_table,
)
from portolan.training.spend import SpendMeter, default_spend_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUTH_TABLE_PATH = PROJECT_ROOT / "corpus" / "truth-table.yaml"

CRM_SOURCE_TO_LOCATOR: dict[str, tuple[str, str]] = {
    "S-crm-A1002": ("crm/accounts.csv", "A-1002"),
    "S-crm-A1010": ("crm/accounts.csv", "A-1010"),
    "S-crm-A1080": ("crm/accounts.csv", "A-1080"),
    "S-crm-opp-A1010": ("crm/opportunities.csv", "O-3001"),
    "S-crm-opp-A1010-v2": ("crm/opportunities-2026-08-21.csv", "O-3001"),
    # v1.1 (added 2026-09-13)
    "S-crm-A1090": ("crm/accounts.csv", "A-1090"),
    "S-crm-A1091": ("crm/accounts.csv", "A-1091"),
}

# --------------------------------------------------------------------------
# Resumability (added 2026-09-13, orchestrator status check after runs 1 and
# 2 of a live full-size generation each crashed partway through on a marker
# miss): `--mode openrouter` calls are real-money, real-wall-clock live model
# calls, and every prior run regenerated EVERY call from scratch on a
# restart even though most of them had already succeeded -- three full
# attempts at the ~24 fact-bearing+noise calls alone is three times the real
# spend for no reason. `_GenerationProgress` is a tiny JSON sidecar next to
# the corpus (`<out_dir>/.generation_progress.json`, never committed --
# see .gitignore) recording, per source id, the exact placements a
# successful `--mode openrouter` generation produced plus this run's
# cumulative OpenRouter spend so far. On the next invocation (crash-restart
# or a deliberate resume), any source whose brief still expects the exact
# same fact ids AND whose output file is still on disk is skipped entirely
# -- its already-written text and previously-recorded placements are reused
# verbatim, no model call made -- and `_check_budget` treats the recorded
# spend as a floor so the $3 cap (`prose.CORPUS_GENERATION_COST_LIMIT_USD`)
# is enforced across restarts, not reset to $0 by every crash.
# --------------------------------------------------------------------------


def _progress_path(out_dir: Path) -> Path:
    return out_dir / ".generation_progress.json"


def _load_progress(out_dir: Path) -> dict[str, Any]:
    path = _progress_path(out_dir)
    if not path.exists():
        return {"spent_usd": 0.0, "sources": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"spent_usd": 0.0, "sources": {}}
    data.setdefault("spent_usd", 0.0)
    data.setdefault("sources", {})
    return data


def _save_progress(out_dir: Path, progress: dict[str, Any]) -> None:
    path = _progress_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    tmp.replace(path)


def _record_source_progress(
    out_dir: Path, progress: dict[str, Any], source_id: str, fact_ids: list[str], placements: list
) -> None:
    progress["sources"][source_id] = {
        "fact_ids": fact_ids,
        "placements": [
            {
                "fact_id": p.fact_id,
                "locator": p.locator,
                "line_or_comment": p.line_or_comment,
                "excerpt": p.excerpt,
            }
            for p in placements
        ],
    }
    _save_progress(out_dir, progress)


def _already_done(
    out_dir: Path,
    progress: dict[str, Any],
    source_id: str,
    expected_fact_ids: list[str],
    out_path: Path,
) -> list[markers.Placement] | None:
    """Placements to reuse, or `None` if `source_id` must be (re)generated:
    not previously recorded, recorded for a DIFFERENT set of expected fact
    ids (the truth table changed under it), or its output file is gone."""
    entry = progress["sources"].get(source_id)
    if entry is None or not out_path.exists():
        return None
    if entry.get("fact_ids") != expected_fact_ids:
        return None
    return [
        markers.Placement(p["fact_id"], p["locator"], p["line_or_comment"], p["excerpt"])
        for p in entry.get("placements", [])
    ]


# The ten new-content snippets for the D03 quoted-thread distractor, oldest
# reply first. Each is prepended above the ever-deepening quote of
# everything before it. No facts are planted here — F21 already lives in
# the original message (S-email-021); this only tests that quoted history
# gets stripped/deduped rather than re-extracted as new facts each reply.
D03_REPLY_BODIES = [
    "Hi Owen, happy to get this made. Can you confirm the shipping address for the printed cards — same as your main facility?",
    "Yes, same address: 214 Aldric Way, main receiving dock. Whenever you have a proof, send it my way.",
    "Proof attached in a separate file — let me know if the translation reads right to your team before we print.",
    "Our clinical educator gave it a look and it reads well. One small note: can 'quick-start' stay in English in the title?",
    "Sure, we can keep 'Quick-Start' in the header and translate the body copy. Updated proof coming shortly.",
    "That version looks good. How many copies should we plan for initially?",
    "Let's start with 150 copies — one per bed plus some spares for the education office.",
    "150 it is. Print run should take about a week and a half once approved.",
    "Sounds good, that timeline works fine on our end. Thanks for turning this around.",
    "Of course — I'll send tracking once they ship. Appreciate you flagging the need.",
]


MAX_GENERATION_ATTEMPTS = 4

_MISSING_MARKER_RE = re.compile(r"^missing marker for (\S+)$")


def _missing_fact_ids(problems: list[str]) -> list[str] | None:
    """If every problem is exactly "missing marker for X", return the list
    of X's; `None` if anything else is wrong (an unexpected/extra marker,
    or N>1 for a fact not in `allow_multiple`) -- those aren't safe for
    `markers.insert_marker_around_best_match`'s assumptions, so the caller
    should fail loudly rather than guess."""
    ids = []
    for p in problems:
        m = _MISSING_MARKER_RE.match(p)
        if not m:
            return None
        ids.append(m.group(1))
    return ids


def _claims_by_id(b: Brief) -> dict[str, str]:
    return {f.fact_id: f.claim for f in b.facts}


def _generate_call_text(
    client: Any, b: Brief, *, out_dir: Path, progress: dict[str, Any]
) -> tuple[str, str, list]:
    """Up to `MAX_GENERATION_ATTEMPTS` tries: `generate_via_openrouter` is a
    live model call, and the one real failure seen in practice (a live
    full-corpus run, 2026-09-13: S-call-008/F17 came back with no marker at
    all) is exactly the kind of stochastic miss a same-prompt retry usually
    fixes, not a deterministic bug worth crashing an hours-long, real-money
    generation run over. If every attempt still comes back missing only
    marker glyphs (not missing the fact itself -- see
    `markers.insert_marker_around_best_match`'s docstring for the exact
    failure this covers, S-email-066/F50, also seen live 2026-09-13), the
    last attempt's own text is repaired instead of thrown away.

    Resumable (see the module's "Resumability" comment above
    `_progress_path`): a prior successful run's already-written output file
    is reused verbatim, no model call made, when `progress` still has a
    matching entry for `b.source_id`."""
    expected = [f.fact_id for f in b.facts]
    out_path = out_dir / "calls" / f"{b.source_id}.txt"
    reused = _already_done(out_dir, progress, b.source_id, expected, out_path)
    if reused is not None:
        return "", out_path.read_text(encoding="utf-8"), reused

    last_problems: list[str] = []
    last_raw = ""
    for _attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        raw = prose.generate_via_openrouter(client, b)
        _check_budget(client, progress["spent_usd"])
        clean_text, placements = markers.process_call(raw)
        problems = markers.verify_exactly_one(
            expected, placements, allow_multiple=MULTI_MENTION_FACTS
        )
        if not problems:
            _record_source_progress(out_dir, progress, b.source_id, expected, placements)
            return raw, clean_text, placements
        last_problems, last_raw = problems, raw

    missing = _missing_fact_ids(last_problems)
    if missing:
        claims = _claims_by_id(b)
        lines = last_raw.splitlines()
        for fact_id in missing:
            repaired = markers.insert_marker_around_best_match(fact_id, claims[fact_id], lines)
            if repaired is not None:
                lines = repaired
        raw = "\n".join(lines)
        clean_text, placements = markers.process_call(raw)
        problems = markers.verify_exactly_one(
            expected, placements, allow_multiple=MULTI_MENTION_FACTS
        )
        if not problems:
            _record_source_progress(out_dir, progress, b.source_id, expected, placements)
            return raw, clean_text, placements
        last_problems = problems
    raise markers.MarkerError(
        f"{b.source_id}: {last_problems} (after {MAX_GENERATION_ATTEMPTS} attempts)"
    )


def _generate_ticket_payload(
    client: Any, b: Brief, *, out_dir: Path, progress: dict[str, Any]
) -> tuple[dict[str, Any], list[dict], list]:
    """Same retry + best-effort-repair + resume rationale as
    `_generate_call_text`, for a ticket brief (repair unit: one comment's
    `body`). The resumed path re-reads the ticket JSON straight off disk
    (already clean -- `structured.write_ticket`'s own output) rather than
    re-deriving comment bodies, since `process_ticket_comments` needs the
    RAW (marker-bearing) comments, which are gone from the written file."""
    import json as _json

    expected = [f.fact_id for f in b.facts]
    out_path = out_dir / "tickets" / f"{b.source_id}.json"
    reused = _already_done(out_dir, progress, b.source_id, expected, out_path)
    if reused is not None:
        payload = _json.loads(out_path.read_text(encoding="utf-8"))
        return payload, payload["comments"], reused

    last_problems: list[str] = []
    last_payload: dict[str, Any] = {}
    for _attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        payload = _parse_generated_ticket_json(prose.generate_via_openrouter(client, b))
        _check_budget(client, progress["spent_usd"])
        clean_comments, placements = markers.process_ticket_comments(
            b.source_id, payload["comments"]
        )
        problems = markers.verify_exactly_one(expected, placements)
        if not problems:
            _record_source_progress(out_dir, progress, b.source_id, expected, placements)
            return payload, clean_comments, placements
        last_problems, last_payload = problems, payload

    missing = _missing_fact_ids(last_problems)
    if missing:
        claims = _claims_by_id(b)
        bodies = [c["body"] for c in last_payload["comments"]]
        for fact_id in missing:
            repaired = markers.insert_marker_around_best_match(fact_id, claims[fact_id], bodies)
            if repaired is not None:
                bodies = repaired
        payload = dict(last_payload)
        payload["comments"] = [
            {**c, "body": body} for c, body in zip(last_payload["comments"], bodies, strict=True)
        ]
        clean_comments, placements = markers.process_ticket_comments(
            b.source_id, payload["comments"]
        )
        problems = markers.verify_exactly_one(expected, placements)
        if not problems:
            _record_source_progress(out_dir, progress, b.source_id, expected, placements)
            return payload, clean_comments, placements
        last_problems = problems
    raise markers.MarkerError(
        f"{b.source_id}: {last_problems} (after {MAX_GENERATION_ATTEMPTS} attempts)"
    )


def _generate_email_body(
    client: Any, b: Brief, *, out_dir: Path, progress: dict[str, Any]
) -> tuple[str, list]:
    """Same retry + best-effort-repair + resume rationale as
    `_generate_call_text`, for an email brief (repair unit: one blank-line-
    separated paragraph -- this is exactly the S-email-066/F50 failure this
    fallback was added for, live 2026-09-13: the model wrote the fact
    plainly, twice, across two separate retries, but never wrapped it in
    marker glyphs). The resumed path reads the already-clean body back out
    of the written `.txt` file (header block, then a blank line, then the
    body) rather than the pre-strip raw text, which was never saved."""
    expected = [f.fact_id for f in b.facts]
    out_path = out_dir / "email" / f"{b.source_id}.txt"
    reused = _already_done(out_dir, progress, b.source_id, expected, out_path)
    if reused is not None:
        _header, _, clean_body = out_path.read_text(encoding="utf-8").partition("\n\n")
        return clean_body.rstrip("\n"), reused

    last_problems: list[str] = []
    last_body = ""
    for _attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        body = prose.generate_via_openrouter(client, b)
        _check_budget(client, progress["spent_usd"])
        clean_body, placements = markers.process_email_body(b.source_id, body)
        problems = markers.verify_exactly_one(expected, placements)
        if not problems:
            _record_source_progress(out_dir, progress, b.source_id, expected, placements)
            return clean_body, placements
        last_problems, last_body = problems, body

    missing = _missing_fact_ids(last_problems)
    if missing:
        claims = _claims_by_id(b)
        paragraphs = last_body.split("\n\n")
        for fact_id in missing:
            repaired = markers.insert_marker_around_best_match(fact_id, claims[fact_id], paragraphs)
            if repaired is not None:
                paragraphs = repaired
        body = "\n\n".join(paragraphs)
        clean_body, placements = markers.process_email_body(b.source_id, body)
        problems = markers.verify_exactly_one(expected, placements)
        if not problems:
            _record_source_progress(out_dir, progress, b.source_id, expected, placements)
            return clean_body, placements
        last_problems = problems
    raise markers.MarkerError(
        f"{b.source_id}: {last_problems} (after {MAX_GENERATION_ATTEMPTS} attempts)"
    )


def _check_budget(client: Any, prior_spend_usd: float = 0.0) -> None:
    """Raise `prose.CorpusBudgetExceeded` the moment cumulative OpenRouter
    cost for this run crosses `prose.CORPUS_GENERATION_COST_LIMIT_USD` --
    checked after every single `generate_via_openrouter` call (not just at
    the end) so a runaway run stops mid-source rather than after finishing
    the whole (expensive) corpus. `prior_spend_usd` (from the resumability
    progress file, `_load_progress`) is added as a floor so the $3 cap is
    enforced across a crash-and-resume, not reset to $0 by every restart."""
    if client is None:
        return
    spent = prior_spend_usd + prose.cumulative_cost_usd(client)
    if spent > prose.CORPUS_GENERATION_COST_LIMIT_USD:
        raise prose.CorpusBudgetExceeded(
            f"OpenRouter cost ${spent:.4f} (including ${prior_spend_usd:.4f} from earlier "
            f"runs) exceeded the ${prose.CORPUS_GENERATION_COST_LIMIT_USD:.2f} budget -- stopping."
        )


def _parse_generated_ticket_json(raw: str) -> dict[str, Any]:
    """`generate_via_openrouter`'s ticket reply should already be one raw
    JSON object (system prompt rule 7), but a model that ignores "no
    markdown code fence" some of the time is a known failure mode of every
    provider this project has benched (`portolan.llm.openrouter`'s own
    module docstring) -- strip a ```json ... ``` (or bare ```) fence before
    parsing rather than let one stray fence lose the whole ticket."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[: -len("```")]
    return json.loads(text.strip())


def bump_quote(text: str) -> str:
    """Add one more level of '>' quoting to every line of `text`."""
    out_lines = []
    for line in text.split("\n"):
        if line.startswith(">"):
            out_lines.append(">" + line)
        elif line == "":
            out_lines.append(">")
        else:
            out_lines.append("> " + line)
    return "\n".join(out_lines)


def estimate_tokens(text: str) -> int:
    return round(len(text.split()) / 0.75)


def _account_ids_for(registry: Registry, participant_ids: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for pid in participant_ids:
        person = registry.person(pid)
        if person.account:
            seen.setdefault(person.account, None)
    return list(seen.keys())


def _parse_staged_email(raw: str) -> tuple[str, str]:
    """Staged email files start with `Subject: ...`, a blank line, then the
    body. Returns (subject, body)."""
    first_line, _, rest = raw.partition("\n")
    subject = first_line.removeprefix("Subject:").strip()
    body = rest.lstrip("\n")
    return subject, body


def process_calls(
    briefs: list[Brief],
    registry: Registry,
    out_dir: Path,
    *,
    mode: str = "manual",
    client: Any = None,
    progress: dict[str, Any] | None = None,
) -> list[manifest_mod.SourceRecord]:
    """`mode="openrouter"` (with `client` from `prose.build_openrouter_client`)
    regenerates EVERY call -- fact-bearing, noise, and distractor alike --
    through the live model at full, un-reduced length, ignoring any staged
    file: this is what makes a full-size run (build instructions item 3,
    "no reduced mode") actually write 12k-25k-token transcripts instead of
    replaying the old manual-mode 6k-9k-token hand-written ones. Every other
    mode keeps the original staged-file-else-filler behavior unchanged."""
    records = []
    for b in briefs:
        if mode == "openrouter" and client is not None:
            _raw, clean_text, placements = _generate_call_text(
                client, b, out_dir=out_dir, progress=progress
            )
        else:
            raw = prose.read_staged(PROJECT_ROOT, b)
            if raw is None:
                raw = prose.generate_filler_call(b)
            clean_text, placements = markers.process_call(raw)
            expected = [f.fact_id for f in b.facts]
            problems = markers.verify_exactly_one(
                expected, placements, allow_multiple=MULTI_MENTION_FACTS
            )
            if problems:
                raise markers.MarkerError(f"{b.source_id}: {problems}")
        out_path = out_dir / "calls" / f"{b.source_id}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(clean_text.rstrip() + "\n", encoding="utf-8")
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="call",
                path=f"calls/{b.source_id}.txt",
                date=b.date,
                participants=[p.id for p in b.participants],
                accounts=b.accounts,
                token_estimate=estimate_tokens(clean_text),
                facts=[
                    manifest_mod.FactPlacement(p.fact_id, p.locator, p.line_or_comment, p.excerpt)
                    for p in placements
                ],
                is_noise=b.is_noise,
                is_distractor=b.is_distractor,
                distractor_id=b.distractor_id,
            )
        )
    return records


def process_tickets(
    briefs: list[Brief],
    registry: Registry,
    out_dir: Path,
    *,
    mode: str = "manual",
    client: Any = None,
    progress: dict[str, Any] | None = None,
) -> list[manifest_mod.SourceRecord]:
    """Tickets have no "reduced" length (GENERATOR.md: 200-1,500 tokens in
    every mode), so a staged file is always reused when present, in every
    mode -- `mode`/`client` only matter for a fact-bearing or distractor
    ticket with NO staged file (every v1.1 ticket), which falls back to a
    live `generate_via_openrouter` call instead of the old always-filler
    behavior (filler carries no facts/markers, so it can't stand in for
    one of these). Noise tickets always use filler, regardless of mode."""
    records = []
    for b in briefs:
        raw_json = prose.read_staged(PROJECT_ROOT, b)
        if raw_json is not None:
            payload = json.loads(raw_json)
            clean_comments, placements = markers.process_ticket_comments(
                b.source_id, payload["comments"]
            )
            expected = [f.fact_id for f in b.facts]
            problems = markers.verify_exactly_one(expected, placements)
            if problems:
                raise markers.MarkerError(f"{b.source_id}: {problems}")
        elif mode == "openrouter" and client is not None and (b.facts or b.is_distractor):
            payload, clean_comments, placements = _generate_ticket_payload(
                client, b, out_dir=out_dir, progress=progress
            )
        else:
            payload = prose.generate_filler_ticket(b)
            clean_comments, placements = markers.process_ticket_comments(
                b.source_id, payload["comments"]
            )
            expected = [f.fact_id for f in b.facts]
            problems = markers.verify_exactly_one(expected, placements)
            if problems:
                raise markers.MarkerError(f"{b.source_id}: {problems}")
        account_name = registry.account(b.accounts[0]).name if b.accounts else ""
        requester_name = registry.person(b.extra["requester"]).name
        out_path = out_dir / "tickets" / f"{b.source_id}.json"
        structured.write_ticket(
            ticket_id=b.source_id,
            requester_name=requester_name,
            account_name=account_name,
            created=b.date,
            subject=payload["subject"],
            comments=clean_comments,
            out_path=out_path,
        )
        total_words = sum(len(c["body"].split()) for c in clean_comments) + len(
            payload["subject"].split()
        )
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="ticket",
                path=f"tickets/{b.source_id}.json",
                date=b.date,
                participants=[b.extra["requester"]],
                accounts=b.accounts,
                token_estimate=round(total_words / 0.75),
                facts=[
                    manifest_mod.FactPlacement(p.fact_id, p.locator, p.line_or_comment, p.excerpt)
                    for p in placements
                ],
                is_noise=b.is_noise,
                is_distractor=b.is_distractor,
                distractor_id=b.distractor_id,
            )
        )
    return records


def process_emails(
    briefs: list[Brief],
    registry: Registry,
    out_dir: Path,
    *,
    mode: str = "manual",
    client: Any = None,
    progress: dict[str, Any] | None = None,
) -> list[manifest_mod.SourceRecord]:
    """Same staged-file-first rule as `process_tickets`: emails have no
    "reduced" length either, so a staged file always wins when present.
    Only a fact-bearing or distractor email with no staged file (every
    v1.1 email) falls back to `generate_via_openrouter`, which -- per the
    system prompt's "emails ONLY" rule -- returns just the body; the
    subject is `b.topic` when the brief set one, else a generic
    "Following up" (matching the existing filler-path default)."""
    records = []
    clean_bodies_by_id: dict[str, str] = {}
    subjects_by_id: dict[str, str] = {}

    # First pass: every email except the D03 thread, which depends on
    # S-email-021's already-cleaned body.
    for b in briefs:
        if b.distractor_id == "D03":
            continue
        raw = prose.read_staged(PROJECT_ROOT, b)
        if raw is not None:
            subject, body = _parse_staged_email(raw)
            clean_body, placements = markers.process_email_body(b.source_id, body)
            expected = [f.fact_id for f in b.facts]
            problems = markers.verify_exactly_one(expected, placements)
            if problems:
                raise markers.MarkerError(f"{b.source_id}: {problems}")
        elif mode == "openrouter" and client is not None and (b.facts or b.is_distractor):
            subject = b.topic or "Following up"
            clean_body, placements = _generate_email_body(
                client, b, out_dir=out_dir, progress=progress
            )
        else:
            subject = b.topic or "Following up"
            body = prose.generate_filler_email_body(b)
            clean_body, placements = markers.process_email_body(b.source_id, body)
            expected = [f.fact_id for f in b.facts]
            problems = markers.verify_exactly_one(expected, placements)
            if problems:
                raise markers.MarkerError(f"{b.source_id}: {problems}")
        clean_bodies_by_id[b.source_id] = clean_body
        subjects_by_id[b.source_id] = subject

        from_id = b.extra["from"]
        to_ids = b.extra["to"]
        from_person = registry.person(from_id)
        to_people = [registry.person(t) for t in to_ids]
        from_addr = (
            f"{from_person.name} <{from_person.email}>" if from_person.email else from_person.name
        )
        to_addrs = [f"{p.name} <{p.email}>" if p.email else p.name for p in to_people]
        file_text = prose.generate_email_file(b, subject, clean_body, from_addr, to_addrs)
        out_path = out_dir / "email" / f"{b.source_id}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(file_text, encoding="utf-8")

        accounts = _account_ids_for(registry, [from_id] + to_ids)
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="email",
                path=f"email/{b.source_id}.txt",
                date=b.date,
                participants=[from_id] + to_ids,
                accounts=accounts,
                token_estimate=estimate_tokens(clean_body),
                facts=[
                    manifest_mod.FactPlacement(p.fact_id, p.locator, p.line_or_comment, p.excerpt)
                    for p in placements
                ],
                is_noise=b.is_noise,
                is_distractor=b.is_distractor,
                distractor_id=b.distractor_id,
            )
        )

    # Second pass: D03, the ten-reply thread quoting S-email-021 in full.
    for b in briefs:
        if b.distractor_id != "D03":
            continue
        original = clean_bodies_by_id["S-email-021"]
        full = original
        for new_content in D03_REPLY_BODIES:
            full = new_content + "\n\n" + bump_quote(full)
        subject = "RE: " + subjects_by_id["S-email-021"]
        from_id = b.extra["from"]
        to_ids = b.extra["to"]
        from_person = registry.person(from_id)
        to_people = [registry.person(t) for t in to_ids]
        from_addr = f"{from_person.name} <{from_person.email}>"
        to_addrs = [f"{p.name} <{p.email}>" for p in to_people]
        file_text = prose.generate_email_file(b, subject, full, from_addr, to_addrs)
        out_path = out_dir / "email" / f"{b.source_id}.txt"
        out_path.write_text(file_text, encoding="utf-8")
        accounts = _account_ids_for(registry, [from_id] + to_ids)
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="email",
                path=f"email/{b.source_id}.txt",
                date=b.date,
                participants=[from_id] + to_ids,
                accounts=accounts,
                token_estimate=estimate_tokens(full),
                facts=[],
                is_noise=False,
                is_distractor=True,
                distractor_id="D03",
            )
        )
    return records


NOISE_RSS_SUMMARIES = [
    "Industry observers note continued interest but no major shifts this cycle.",
    "The post drew a handful of comments from local health-system staff.",
    "No material developments reported beyond routine community engagement.",
]


def process_rss(briefs: list[Brief], out_dir: Path) -> list[manifest_mod.SourceRecord]:
    records = []
    industry_items: list[dict[str, Any]] = []
    larkspur_items: list[dict[str, Any]] = []

    PLANTED_RSS: dict[str, dict[str, str]] = {
        "S-rss-002": {
            "title": "Riverside Clinic in Bend, Oregon switches infusion pump vendor",
            "summary": (
                "Riverside Clinic, an independent outpatient surgical clinic in Bend, Oregon, "
                "has adopted a competing infusion pump line for its recovery bays, clinic "
                "administrators said this week. The clinic is not affiliated with any Midwest "
                "or Northeast hospital system."
            ),
        },
        "S-rss-004": {
            "title": "Corvane Systems recalls C-40 pump line",
            "summary": (
                "Corvane Systems issued a voluntary recall of its C-40 infusion pump line on "
                "2026-08-26, citing a software defect identified during post-market surveillance. "
                "The company said replacement units are being prioritized for affected hospitals."
            ),
        },
        "S-rss-006": {
            "title": "Larkspur Regional Health names new CFO",
            "summary": (
                "Larkspur Regional Health announced Renata Solis as its new Chief Financial "
                "Officer, effective September 1, 2026, succeeding the system's outgoing finance "
                "lead as part of a planned transition."
            ),
        },
        # v1.1 (added 2026-09-13)
        "S-rss-007": {
            "title": "Northlake Health System promotes Derek Hollis to VP of Facilities",
            "summary": (
                "Northlake Health System announced that Derek Hollis, formerly its Director of "
                "Clinical Engineering, has been promoted to Vice President of Facilities, "
                "effective September 2026."
            ),
        },
    }

    rng = random.Random("rss-noise")
    for b in briefs:
        feed = b.extra["feed"]
        link = structured.rss_link(b.source_id)
        if b.source_id in PLANTED_RSS:
            title = PLANTED_RSS[b.source_id]["title"]
            summary = PLANTED_RSS[b.source_id]["summary"]
        else:
            title = b.topic or "Industry update"
            summary = rng.choice(NOISE_RSS_SUMMARIES)
        item = {"id": b.source_id, "date": b.date, "title": title, "link": link, "summary": summary}
        (industry_items if feed == "industry-news" else larkspur_items).append(item)

        placements = []
        if b.facts:
            fact_id = b.facts[0].fact_id
            locator = structured.rss_locator(link, b.date)
            placements.append(
                manifest_mod.FactPlacement(fact_id, locator, b.source_id, summary[:160])
            )
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="rss",
                path=f"rss/{'industry-news' if feed == 'industry-news' else 'linkedin-larkspur'}.json",
                date=b.date,
                participants=[],
                accounts=[],
                token_estimate=estimate_tokens(summary),
                facts=placements,
                is_noise=b.is_noise,
                is_distractor=b.is_distractor,
                distractor_id=b.distractor_id,
            )
        )

    industry_items.sort(key=lambda i: i["date"])
    larkspur_items.sort(key=lambda i: i["date"])
    structured.write_rss_feed(industry_items, out_dir / "rss" / "industry-news.json")
    structured.write_rss_feed(larkspur_items, out_dir / "rss" / "linkedin-larkspur.json")
    return records


def process_crm(
    briefs: list[Brief], truth: dict[str, Any], out_dir: Path
) -> list[manifest_mod.SourceRecord]:
    structured.write_crm(truth, out_dir / "crm")
    records = []
    for b in briefs:
        path, crm_id = CRM_SOURCE_TO_LOCATOR[b.source_id]
        locator = structured.crm_locator(path, crm_id)
        placements = [
            manifest_mod.FactPlacement(f.fact_id, locator, crm_id, f.claim[:160]) for f in b.facts
        ]
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="crm",
                path=path,
                date=b.date,
                participants=[],
                accounts=[],
                token_estimate=0,
                facts=placements,
            )
        )
    return records


def _final_source_text(brief: Brief, out_dir: Path) -> str | None:
    if brief.source_type == "call":
        p = out_dir / "calls" / f"{brief.source_id}.txt"
    elif brief.source_type == "ticket":
        p = out_dir / "tickets" / f"{brief.source_id}.json"
    elif brief.source_type == "email":
        p = out_dir / "email" / f"{brief.source_id}.txt"
    else:
        return None
    return p.read_text(encoding="utf-8") if p.exists() else None


def check_forbidden_words(briefs: list[Brief], out_dir: Path) -> list[str]:
    """Case-insensitive check that no fact's forbidden word/phrase (a
    branch name, "risk"/"churn", a competitor name, etc.) leaked into the
    final, marker-stripped text of the source that plants it. Separate from
    `manifest.validate()`, which only checks placement sets and locator
    formats — this checks the prose itself, staged content included."""
    problems: list[str] = []
    for b in briefs:
        forbidden = {phrase for f in b.facts for phrase in f.forbidden}
        if not forbidden:
            continue
        text = _final_source_text(b, out_dir)
        if text is None:
            continue
        lowered = text.lower()
        for phrase in forbidden:
            if phrase.lower() in lowered:
                problems.append(f"{b.source_id}: forbidden phrase {phrase!r} found in final text")
    return problems


def generate(
    out_dir: Path, mode_arg: str
) -> tuple[dict[str, Any], list[str], float, dict[str, int]]:
    """Returns (manifest, problems, cost_usd, usage_totals). `cost_usd` and
    `usage_totals` (`{"calls": n, "input_tokens": n, "output_tokens": n}`,
    OpenRouter calls only) are 0/empty for every mode except "openrouter"."""
    truth = load_truth_table(TRUTH_TABLE_PATH)
    registry = Registry(truth)

    if mode_arg == "auto":
        mode, _reason = prose.detect_mode(PROJECT_ROOT)
    else:
        mode, _reason = mode_arg, "forced by --mode"
    reduced = mode == "manual"

    client: Any = None
    progress: dict[str, Any] = {"spent_usd": 0.0, "sources": {}}
    if mode == "openrouter":
        client = prose.build_openrouter_client(PROJECT_ROOT)
        progress = _load_progress(out_dir)

    briefs = build_briefs(truth, mode, reduced)  # type: ignore[arg-type]
    by_type: dict[str, list[Brief]] = {"call": [], "ticket": [], "email": [], "rss": [], "crm": []}
    for b in briefs:
        by_type[b.source_type].append(b)

    records: list[manifest_mod.SourceRecord] = []
    try:
        records += process_calls(
            by_type["call"], registry, out_dir, mode=mode, client=client, progress=progress
        )
        records += process_tickets(
            by_type["ticket"], registry, out_dir, mode=mode, client=client, progress=progress
        )
        records += process_emails(
            by_type["email"], registry, out_dir, mode=mode, client=client, progress=progress
        )
        records += process_rss(by_type["rss"], out_dir)
        records += process_crm(by_type["crm"], truth, out_dir)
    finally:
        live_cost = prose.cumulative_cost_usd(client) if client is not None else 0.0
        cost_usd = progress["spent_usd"] + live_cost if client is not None else 0.0
        if client is not None:
            # Persist this run's own spend into the floor `_check_budget`
            # reads next time, so a later restart's cap check starts from
            # the true across-runs total, not $0.
            progress["spent_usd"] = cost_usd
            _save_progress(out_dir, progress)
        usage_totals = {
            "calls": len(client.sink.events) if client is not None else 0,
            "input_tokens": sum(e.input_tokens for e in client.sink.events) if client else 0,
            "output_tokens": sum(e.output_tokens for e in client.sink.events) if client else 0,
        }

    manifest = manifest_mod.build_manifest(
        mode=mode,
        reduced=reduced,
        generated_at=_dt.datetime.now(_dt.UTC).isoformat(),
        truth_table_path=TRUTH_TABLE_PATH,
        sources=records,
    )
    manifest_mod.write_manifest(manifest, out_dir / "manifest.json")
    problems = manifest_mod.validate(manifest, truth)
    problems += check_forbidden_words(briefs, out_dir)
    if mode == "openrouter" and not problems:
        # Full success: the progress sidecar has done its job (every source
        # is now in the real manifest/corpus), so remove it rather than
        # leave a stale file that would silently short-circuit a future,
        # deliberate regeneration.
        _progress_path(out_dir).unlink(missing_ok=True)
    return manifest, problems, cost_usd, usage_totals


# ==========================================================================
# `synth` subcommand — synthetic-data pipeline, added 2026-09-19.
#
# Builds a FRESH, fully-synthetic truth table from a seed
# (`corpus.generator.worlds.generate_world`) — never touches
# `corpus/truth-table.yaml` or `corpus/out/` — and writes it as one
# self-contained batch: `calls/`, `email/`, `tickets/`, `crm/`, `slack/`,
# `teams/` under `--out`, plus `manifest.json` and `truth-table.yaml`
# beside them. Prose comes from a ROTATION of FREE OpenRouter models
# (`prose.discover_free_models`/`TeacherRotation`) under a shared
# token-bucket rate limit (`prose.TokenBucketLimiter`) — never the paid
# `CORPUS_WRITER_MODEL` the `generate` subcommand above uses, and never a
# request that would cost money (build instructions: "Free models only").
#
# See `01-spec/synthetic-data-pipeline-2026-09-19.md` for the design,
# the teacher list a real run actually used, and that run's numbers.
# ==========================================================================


# Paid-teacher defaults (added 2026-09-22) -- same 60 requests/minute
# convention `scripts/data/silver_label.py`'s `PAID_TEACHER_RPM` already
# uses for paid OpenRouter models (no published fixed number; a
# conservative default, not a header-derived one) and the same
# `requests_per_day=10**9` trick that module uses to make
# `TokenBucketLimiter`'s daily window never fire for a paid run (there is
# no free-tier daily budget to enforce once every teacher is paid).
PAID_TEACHER_RPM = 60
PAID_TEACHER_RPD = 10**9

DEFAULT_PAID_BUDGET_USD = 5.00

# The `portolan.training.spend.SpendMeter` corpus key every paid-teacher
# synth run records under and checks `--budget-usd` against (fixed
# 2026-09-22: the check used to compare the meter's shared `total_usd` --
# every corpus combined, including a concurrent silver-labeling run's
# spend -- instead of this one; see `SpendMeter.is_corpus_over_budget`).
# One constant so `RunState.corpus_key`'s default, `_synth_generate_unit`'s
# check, and `scripts/data/synth_batches.py`'s run-start/per-world/stop
# printing all agree on the same string.
SYNTH_CORPUS_KEY = "synthetic"


def _synth_progress_path(out_dir: Path) -> Path:
    return out_dir / ".synth_progress.json"


def _synth_load_progress(out_dir: Path) -> dict[str, Any]:
    path = _synth_progress_path(out_dir)
    if not path.exists():
        return {"sources": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"sources": {}}
    data.setdefault("sources", {})
    return data


def _synth_save_progress(out_dir: Path, progress: dict[str, Any]) -> None:
    path = _synth_progress_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass
class SynthUnit:
    """One generated prose unit — a call transcript, a ticket's parsed
    payload, an email body, one chat message, or one CRM note body —
    always with markers already stripped (`clean`), plus every placement
    `process_fn` found (`markers.Placement`, in document order).

    For `calls`/`tickets`/`emails`/CRM notes, `placements[i].locator` is
    already the real, final locator (one source = one file, so
    `markers.process_call`/etc. compute it directly). For `slack`/`teams`,
    the caller's `process_fn` fills `locator` with a placeholder (the real
    one needs a whole channel/day group assembled first — build
    instructions: "locators the manifest can record ... channel, date,
    message index") and the chat-assembly step below replaces it."""

    clean: Any
    placements: list[markers.Placement]
    model: str | None
    failed: bool

    @property
    def fact_excerpts(self) -> dict[str, str]:
        return {p.fact_id: p.excerpt for p in self.placements}


@dataclass
class RunState:
    """Shared, mutable run-wide state threaded through every per-kind
    processing function. `stopped`/`stop_reason` let one kind's loop bail
    out of `prose.DailyLimitExceeded` without raising past this module's
    `synth()` — the batch keeps whatever it already generated (`generate`
    incrementally records to `_synth_progress_path` after every unit) and
    every later kind is skipped rather than attempted and failed the same
    way. `teacher_log["failures"]` is the "recorded as failed" list build
    instructions item 2 asks for."""

    stopped: bool = False
    stop_reason: str = ""
    verified: int = 0
    filler_used: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    # Paid-teacher spend tracking (added 2026-09-22) -- `spend` is `None`
    # for every free-teacher run (the default), the unchanged path. When
    # `synth()` is given `teachers=[...paid ids...]`, it sets `spend` to a
    # shared `portolan.training.spend.SpendMeter` and `budget_usd` to the
    # run's dollar ceiling; `_synth_generate_unit` checks `spend.
    # is_corpus_over_budget(corpus_key, budget_usd)` before every live call
    # -- THIS corpus's spend only, not the meter's shared total, which may
    # include an unrelated concurrent run's spend -- and records each
    # call's real cost into it under `corpus_key`.
    spend: Any = None
    budget_usd: float = 0.0
    corpus_key: str = SYNTH_CORPUS_KEY


def _placement_to_dict(p: markers.Placement) -> dict[str, str]:
    return {
        "fact_id": p.fact_id,
        "locator": p.locator,
        "line_or_comment": p.line_or_comment,
        "excerpt": p.excerpt,
    }


def _placement_from_dict(d: dict[str, str]) -> markers.Placement:
    return markers.Placement(d["fact_id"], d["locator"], d["line_or_comment"], d["excerpt"])


def _record_synth_progress(
    out_dir: Path, progress: dict[str, Any], sid: str, expected_fact_ids: list[str], unit: SynthUnit
) -> None:
    progress["sources"][sid] = {
        "fact_ids": expected_fact_ids,
        "clean": unit.clean,
        "placements": [_placement_to_dict(p) for p in unit.placements],
        "model": unit.model,
        "failed": unit.failed,
    }
    _synth_save_progress(out_dir, progress)


def _synth_reuse(
    progress: dict[str, Any], sid: str, expected_fact_ids: list[str]
) -> SynthUnit | None:
    entry = progress["sources"].get(sid)
    if entry is None or entry.get("fact_ids") != expected_fact_ids:
        return None
    placements = [_placement_from_dict(p) for p in entry.get("placements", [])]
    return SynthUnit(entry["clean"], placements, entry.get("model"), entry.get("failed", False))


def _synth_generate_unit(
    brief: Brief,
    *,
    client: Any,
    rotation: prose.TeacherRotation,
    limiter: prose.TokenBucketLimiter,
    system_prompt: str,
    process_fn: Callable[[Any], tuple[Any, list[markers.Placement]]],
    filler_fn: Callable[[Brief], Any],
    out_dir: Path,
    progress: dict[str, Any],
    state: RunState,
    allow_multiple: frozenset[str] = frozenset(),
    locator_pattern: re.Pattern[str] | None = None,
) -> SynthUnit:
    """Generate one prose unit for `brief`.

    - Resumable: a prior run's already-verified (or already-recorded-
      failed) unit for this exact fact-id set is reused verbatim, no model
      call made.
    - Noise (`not brief.facts`): deterministic filler, no network call —
      same "don't spend free-tier budget on text nothing will ever score"
      rationale as `prose.py`'s existing call/ticket/email fillers.
    - Fact-bearing: up to TWO live attempts, each on a DIFFERENT teacher
      (build instructions item 2: "retried once with a different
      teacher"), verified with `markers.verify_exactly_one` AND, when
      `locator_pattern` is given, every placement's locator against it.
      The locator check exists because `verify_exactly_one` only checks
      fact-id presence/count, not locator well-formedness -- a real batch
      run (2026-09-19) caught a teacher that wrote a call line with no
      `Speaker:` prefix at all, which `markers.process_call`'s `CALL_LINE_RE`
      then silently reads as `ts=""` (empty locator): the fact WAS present
      (verify_exactly_one saw no problem) but the placement was unusable.
      Checking the locator here means that exact failure now retries on a
      different teacher instead of being accepted as "verified" with a
      broken locator. Both failing is recorded into `state.failures` and
      this source falls back to filler text (so the corpus still has a
      complete, readable file for every declared source) with `failed=True`
      and no fact excerpt — the resulting manifest will show that fact's
      placements short of what the truth table expects, which is the
      honest outcome, not silently invented.
    - `prose.DailyLimitExceeded` is never swallowed: it propagates to the
      caller, which sets `state.stopped` and stops the batch. Paid-teacher
      runs (`state.spend is not None`) use the SAME propagation path via
      `prose.PaidBudgetExceeded` (a `DailyLimitExceeded` subclass, raised
      here before the next live attempt once `state.spend.
      is_corpus_over_budget(state.corpus_key, state.budget_usd)` -- THIS
      run's own `corpus_key` spend, not the shared meter's grand total,
      which a concurrent run against a different corpus can inflate), and
      every real paid call's cost is recorded into `state.spend` right
      after it returns (`client.sink.events`, the same `cost_estimate`
      delta `portolan.training.pipeline._sink_cost_delta` reads for the
      silver-labeling pipeline).
    """
    expected = [f.fact_id for f in brief.facts]
    reused = _synth_reuse(progress, brief.source_id, expected)
    if reused is not None:
        return reused

    if not brief.facts:
        raw = filler_fn(brief)
        clean, _placements = process_fn(raw)
        unit = SynthUnit(clean, [], None, False)
        _record_synth_progress(out_dir, progress, brief.source_id, expected, unit)
        return unit

    last_model: str | None = None
    last_problems: list[str] = ["no attempt made"]
    for attempt in range(2):
        if state.spend is not None and state.spend.is_corpus_over_budget(
            state.corpus_key, state.budget_usd
        ):
            raise prose.PaidBudgetExceeded(
                f"{prose.PAID_BUDGET_STOP_MARKER}: "
                f"{state.corpus_key} spend "
                f"${state.spend.corpus_usd(state.corpus_key):.4f} >= ${state.budget_usd:.2f} budget-usd"
            )
        events_before = len(client.sink.events) if state.spend is not None else 0
        try:
            raw, model = prose.generate_via_free_teacher(
                client, brief, rotation, limiter, system=system_prompt, exclude_model=last_model
            )
        except prose.DailyLimitExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - a flaky free model, retried once
            last_problems = [f"generation error (attempt {attempt + 1}/2): {exc}"]
            last_model = None
            continue
        if state.spend is not None:
            cost = sum((e.cost_estimate or 0.0) for e in client.sink.events[events_before:])
            if cost > 0:
                state.spend.record(model=model, corpus=state.corpus_key, amount_usd=cost)
        try:
            clean, placements = process_fn(raw)
        except Exception as exc:  # noqa: BLE001 - e.g. a ticket that wasn't valid JSON
            last_problems = [f"parse error (attempt {attempt + 1}/2): {exc}"]
            last_model = model
            continue
        problems = markers.verify_exactly_one(expected, placements, allow_multiple=allow_multiple)
        if not problems and locator_pattern is not None:
            problems = [
                f"invalid locator for {p.fact_id}: {p.locator!r} does not match "
                f"{locator_pattern.pattern!r}"
                for p in placements
                if not locator_pattern.match(p.locator)
            ]
        if not problems:
            unit = SynthUnit(clean, placements, model, False)
            _record_synth_progress(out_dir, progress, brief.source_id, expected, unit)
            state.verified += 1
            return unit
        last_model, last_problems = model, problems

    state.failures.append({"source_id": brief.source_id, "problems": last_problems})
    raw = filler_fn(brief)
    clean, _placements = process_fn(raw)
    unit = SynthUnit(clean, [], None, True)
    _record_synth_progress(out_dir, progress, brief.source_id, expected, unit)
    state.filler_used += 1
    return unit


def _synth_placements_to_facts(unit: SynthUnit) -> list[manifest_mod.FactPlacement]:
    return [
        manifest_mod.FactPlacement(p.fact_id, p.locator, p.line_or_comment, p.excerpt)
        for p in unit.placements
    ]


def _synth_process_calls(
    briefs_list: list[Brief],
    out_dir: Path,
    *,
    client: Any,
    rotation: prose.TeacherRotation,
    limiter: prose.TokenBucketLimiter,
    progress: dict[str, Any],
    state: RunState,
) -> list[manifest_mod.SourceRecord]:
    records: list[manifest_mod.SourceRecord] = []
    for b in briefs_list:
        if state.stopped:
            break
        try:
            unit = _synth_generate_unit(
                b,
                client=client,
                rotation=rotation,
                limiter=limiter,
                system_prompt=prose.STABLE_SYSTEM_PROMPT,
                process_fn=markers.process_call,
                filler_fn=prose.generate_filler_call,
                out_dir=out_dir,
                progress=progress,
                state=state,
                allow_multiple=MULTI_MENTION_FACTS,
                locator_pattern=manifest_mod.LOCATOR_PATTERNS["call"],
            )
        except prose.DailyLimitExceeded as exc:
            state.stopped, state.stop_reason = True, str(exc)
            break
        clean_text = str(unit.clean)
        out_path = out_dir / "calls" / f"{b.source_id}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(clean_text.rstrip() + "\n", encoding="utf-8")
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="call",
                path=f"calls/{b.source_id}.txt",
                date=b.date,
                participants=[p.id for p in b.participants],
                accounts=b.accounts,
                token_estimate=estimate_tokens(clean_text),
                facts=_synth_placements_to_facts(unit),
                is_noise=b.is_noise,
                model=unit.model,
            )
        )
    return records


def _synth_process_tickets(
    briefs_list: list[Brief],
    registry: Registry,
    out_dir: Path,
    *,
    client: Any,
    rotation: prose.TeacherRotation,
    limiter: prose.TokenBucketLimiter,
    progress: dict[str, Any],
    state: RunState,
) -> list[manifest_mod.SourceRecord]:
    def process_fn(raw: Any) -> tuple[dict[str, Any], list[markers.Placement]]:
        payload = raw if isinstance(raw, dict) else _parse_generated_ticket_json(raw)
        clean_comments, placements = markers.process_ticket_comments(
            b.source_id, payload["comments"]
        )
        return {"subject": payload["subject"], "comments": clean_comments}, placements

    records: list[manifest_mod.SourceRecord] = []
    for b in briefs_list:
        if state.stopped:
            break
        try:
            unit = _synth_generate_unit(
                b,
                client=client,
                rotation=rotation,
                limiter=limiter,
                system_prompt=prose.STABLE_SYSTEM_PROMPT,
                process_fn=process_fn,
                filler_fn=prose.generate_filler_ticket,
                out_dir=out_dir,
                progress=progress,
                state=state,
                locator_pattern=manifest_mod.LOCATOR_PATTERNS["ticket"],
            )
        except prose.DailyLimitExceeded as exc:
            state.stopped, state.stop_reason = True, str(exc)
            break
        account_name = registry.account(b.accounts[0]).name if b.accounts else ""
        requester_name = registry.person(b.extra["requester"]).name
        out_path = out_dir / "tickets" / f"{b.source_id}.json"
        structured.write_ticket(
            ticket_id=b.source_id,
            requester_name=requester_name,
            account_name=account_name,
            created=b.date,
            subject=unit.clean["subject"],
            comments=unit.clean["comments"],
            out_path=out_path,
        )
        total_words = sum(len(c["body"].split()) for c in unit.clean["comments"]) + len(
            unit.clean["subject"].split()
        )
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="ticket",
                path=f"tickets/{b.source_id}.json",
                date=b.date,
                participants=[b.extra["requester"]],
                accounts=b.accounts,
                token_estimate=round(total_words / 0.75),
                facts=_synth_placements_to_facts(unit),
                is_noise=b.is_noise,
                model=unit.model,
            )
        )
    return records


def _synth_process_emails(
    briefs_list: list[Brief],
    registry: Registry,
    out_dir: Path,
    *,
    client: Any,
    rotation: prose.TeacherRotation,
    limiter: prose.TokenBucketLimiter,
    progress: dict[str, Any],
    state: RunState,
) -> list[manifest_mod.SourceRecord]:
    def process_fn(raw: Any) -> tuple[str, list[markers.Placement]]:
        return markers.process_email_body(b.source_id, str(raw))

    records: list[manifest_mod.SourceRecord] = []
    for b in briefs_list:
        if state.stopped:
            break
        try:
            unit = _synth_generate_unit(
                b,
                client=client,
                rotation=rotation,
                limiter=limiter,
                system_prompt=prose.STABLE_SYSTEM_PROMPT,
                process_fn=process_fn,
                filler_fn=prose.generate_filler_email_body,
                out_dir=out_dir,
                progress=progress,
                state=state,
                locator_pattern=manifest_mod.LOCATOR_PATTERNS["email"],
            )
        except prose.DailyLimitExceeded as exc:
            state.stopped, state.stop_reason = True, str(exc)
            break
        subject = b.topic or "Following up"
        from_id = b.extra["from"]
        to_ids = b.extra["to"]
        from_person = registry.person(from_id)
        to_people = [registry.person(t) for t in to_ids]
        from_addr = (
            f"{from_person.name} <{from_person.email}>" if from_person.email else from_person.name
        )
        to_addrs = [f"{p.name} <{p.email}>" if p.email else p.name for p in to_people]
        file_text = prose.generate_email_file(b, subject, str(unit.clean), from_addr, to_addrs)
        out_path = out_dir / "email" / f"{b.source_id}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(file_text, encoding="utf-8")
        accounts = _account_ids_for(registry, [from_id] + to_ids)
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="email",
                path=f"email/{b.source_id}.txt",
                date=b.date,
                participants=[from_id] + to_ids,
                accounts=accounts,
                token_estimate=estimate_tokens(str(unit.clean)),
                facts=_synth_placements_to_facts(unit),
                is_noise=b.is_noise,
                model=unit.model,
            )
        )
    return records


def _synth_process_crm_notes(
    briefs_list: list[Brief],
    truth: dict[str, Any],
    registry: Registry,
    out_dir: Path,
    *,
    client: Any,
    rotation: prose.TeacherRotation,
    limiter: prose.TokenBucketLimiter,
    progress: dict[str, Any],
    state: RunState,
) -> list[manifest_mod.SourceRecord]:
    contact_crm_id_by_id = {c["id"]: c.get("crm_id") for c in truth["contacts"]}
    records: list[manifest_mod.SourceRecord] = []
    for b in briefs_list:
        if state.stopped:
            break
        note_id = b.extra["note_id"]
        relpath = structured.activity_note_relpath(note_id)

        def process_fn(raw: Any, relpath: str = relpath) -> tuple[str, list[markers.Placement]]:
            paragraphs = str(raw).split("\n\n")
            clean_paragraphs, placements = markers.process_indexed_texts(
                paragraphs, lambda idx: f"{relpath}#p{idx}"
            )
            return "\n\n".join(clean_paragraphs), placements

        try:
            unit = _synth_generate_unit(
                b,
                client=client,
                rotation=rotation,
                limiter=limiter,
                system_prompt=prose.NOTE_SYSTEM_PROMPT,
                process_fn=process_fn,
                filler_fn=lambda brief: prose.generate_filler_note_body(brief.source_id),
                out_dir=out_dir,
                progress=progress,
                state=state,
                locator_pattern=manifest_mod.LOCATOR_PATTERNS["crm"],
            )
        except prose.DailyLimitExceeded as exc:
            state.stopped, state.stop_reason = True, str(exc)
            break
        account_id = b.extra.get("account_id")
        account_crm_id = registry.account(account_id).crm_id if account_id else ""
        contact_id = b.extra.get("contact_id")
        contact_crm_id = contact_crm_id_by_id.get(contact_id) if contact_id else None
        out_path = structured.activity_note_path(out_dir, note_id)
        structured.write_activity_note(
            note_id=note_id,
            account_crm_id=account_crm_id,
            contact_crm_id=contact_crm_id,
            author=b.extra["author_id"],
            created_date=b.date,
            subject="Account touchpoint",
            body=str(unit.clean),
            out_path=out_path,
        )
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="crm",
                path=relpath,
                date=b.date,
                participants=[b.extra["author_id"]],
                accounts=[b.extra["account_id"]] if b.extra.get("account_id") else [],
                token_estimate=estimate_tokens(str(unit.clean)),
                facts=_synth_placements_to_facts(unit),
                is_noise=b.is_noise,
                model=unit.model,
            )
        )
    return records


def _synth_process_chat(
    kind: str,
    briefs_list: list[Brief],
    out_dir: Path,
    *,
    client: Any,
    rotation: prose.TeacherRotation,
    limiter: prose.TokenBucketLimiter,
    progress: dict[str, Any],
    state: RunState,
    bot_user_id: str,
    bot_display_name: str,
) -> tuple[list[manifest_mod.SourceRecord], dict[str, str]]:
    """Shared Slack/Teams processing: generate every message (teacher
    rotation for fact-bearing ones, filler for noise), then group by
    (channel, date) IN THE ORDER `truth["sources"][kind]["items"]` already
    lists them and assign each message its 1-based position in that
    group's file -- the "message index" half of the locator (build
    instructions item 3's chat locator format; see
    `structured.slack_locator`/`teams_locator`). Returns `(records,
    channel_to_account)` -- the second only meaningful for `kind=="slack"`,
    used to write `channels.json`.

    `bot_user_id`/`bot_display_name` name this world's OWN support bot
    (derived by the caller from `truth["meta"]["supplier"]["short"]`, added
    2026-09-21 for "distinct worlds" -- see `worlds.py`'s docstring) rather
    than a fixed `"bot-halvard-support"`/`"Halvard Support"`."""

    def process_fn(raw: Any) -> tuple[str, list[markers.Placement]]:
        clean_list, placements = markers.process_indexed_texts([str(raw)], lambda _idx: "")
        return clean_list[0], placements

    def filler_fn(brief: Brief) -> str:
        return prose.generate_filler_chat_message(
            brief.source_id, bool(brief.extra.get("is_customer", True))
        )

    units: dict[str, SynthUnit] = {}
    for b in briefs_list:
        if state.stopped:
            units[b.source_id] = SynthUnit("", [], None, True)
            continue
        try:
            units[b.source_id] = _synth_generate_unit(
                b,
                client=client,
                rotation=rotation,
                limiter=limiter,
                system_prompt=prose.CHAT_SYSTEM_PROMPT,
                process_fn=process_fn,
                filler_fn=filler_fn,
                out_dir=out_dir,
                progress=progress,
                state=state,
            )
        except prose.DailyLimitExceeded as exc:
            state.stopped, state.stop_reason = True, str(exc)
            units[b.source_id] = SynthUnit("", [], None, True)

    groups: dict[tuple[str, str], list[Brief]] = {}
    for b in briefs_list:
        groups.setdefault((b.extra["channel"], b.date), []).append(b)

    records: list[manifest_mod.SourceRecord] = []
    channel_to_account: dict[str, str] = {}
    id_counter = 1_700_000_000_000

    for (channel, day), group_briefs in groups.items():
        messages: list[dict[str, Any]] = []
        for b in group_briefs:
            unit = units[b.source_id]
            user_ref = b.participants[0].id if b.participants else "unknown"
            display_name = b.participants[0].name if b.participants else "Unknown"
            index = len(messages) + 1

            if kind == "slack":
                messages.append(
                    {
                        "type": "message",
                        "user": user_ref,
                        "text": str(unit.clean),
                        "ts": structured.slack_ts(day, index * 300),
                        "thread_ts": None,
                    }
                )
                locator = structured.slack_locator(channel, day, index)
            else:
                id_counter += 1
                messages.append(
                    {
                        "id": str(id_counter),
                        "from": {"user": {"displayName": display_name}},
                        "body": {"content": str(unit.clean)},
                        "createdDateTime": structured.teams_created_datetime(day, index * 300),
                        "replyToId": None,
                    }
                )
                locator = structured.teams_locator(channel, day, index)
            root_msg = messages[-1]

            facts = [
                manifest_mod.FactPlacement(p.fact_id, locator, str(index), p.excerpt)
                for p in unit.placements
            ]
            records.append(
                manifest_mod.SourceRecord(
                    id=b.source_id,
                    type=kind,
                    path=(
                        structured.slack_channel_path(channel, day)
                        if kind == "slack"
                        else structured.teams_channel_path(channel, day)
                    ),
                    date=b.date,
                    participants=[p.id for p in b.participants],
                    accounts=b.accounts,
                    token_estimate=estimate_tokens(str(unit.clean)),
                    facts=facts,
                    is_noise=b.is_noise,
                    model=unit.model,
                )
            )

            if b.extra.get("thread") and b.facts and not unit.failed:
                reply_text = prose.generate_filler_chat_reply(b.source_id)
                reply_index = len(messages) + 1
                if kind == "slack":
                    messages.append(
                        {
                            "type": "message",
                            "user": bot_user_id,
                            "text": reply_text,
                            "ts": structured.slack_ts(day, reply_index * 300),
                            "thread_ts": root_msg["ts"],
                        }
                    )
                else:
                    id_counter += 1
                    messages.append(
                        {
                            "id": str(id_counter),
                            "from": {"user": {"displayName": bot_display_name}},
                            "body": {"content": reply_text},
                            "createdDateTime": structured.teams_created_datetime(
                                day, reply_index * 300
                            ),
                            "replyToId": root_msg["id"],
                        }
                    )

            if b.accounts:
                channel_to_account.setdefault(channel, b.accounts[0])

        if kind == "slack":
            structured.write_slack_day(
                messages, out_dir / structured.slack_channel_path(channel, day)
            )
        else:
            structured.write_teams_day(
                messages, out_dir / structured.teams_channel_path(channel, day)
            )

    return records, channel_to_account


def _synth_process_crm_structured(
    truth: dict[str, Any], registry: Registry, out_dir: Path
) -> list[manifest_mod.SourceRecord]:
    """Fully structured — no model call. Writes `crm/accounts.csv`,
    `crm/contacts.csv` (via the existing, already-generic `structured.
    write_accounts_csv`/`write_contacts_csv`), plus the richer
    `crm/opportunities.csv`/`crm/cases.csv` (`structured.
    write_synth_opportunities_csv`/`write_synth_cases_csv`, build
    instructions item 3). Manifest records for the CRM sources the truth
    table declared (`sources.crm.items`, each carrying its own `path`/
    `crm_id` -- see `worlds.py` -- so no hardcoded id->locator table is
    needed here, unlike `CRM_SOURCE_TO_LOCATOR` above for the original
    corpus)."""
    out_dir_crm = out_dir / "crm"
    structured.write_accounts_csv(truth, out_dir_crm / "accounts.csv")
    structured.write_contacts_csv(truth, out_dir_crm / "contacts.csv")
    crm_records = truth.get("crm_records", {})
    structured.write_synth_opportunities_csv(
        crm_records.get("opportunities", []), out_dir_crm / "opportunities.csv"
    )
    structured.write_synth_cases_csv(crm_records.get("cases", []), out_dir_crm / "cases.csv")

    records: list[manifest_mod.SourceRecord] = []
    for b in build_crm(truth, registry):
        item = next(i for i in truth["sources"]["crm"]["items"] if i["id"] == b.source_id)
        locator = structured.crm_locator(item["path"], item["crm_id"])
        records.append(
            manifest_mod.SourceRecord(
                id=b.source_id,
                type="crm",
                path=item["path"],
                date=b.date,
                participants=[],
                accounts=[],
                token_estimate=0,
                facts=[
                    manifest_mod.FactPlacement(f.fact_id, locator, item["crm_id"], f.claim[:160])
                    for f in b.facts
                ],
            )
        )
    return records


def synth(
    seed: int,
    num_accounts: int,
    per_kind: int,
    out_dir: Path,
    *,
    requests_per_minute: int = 20,
    requests_per_day: int = 1000,
    type_mix: dict[str, int] | None = None,
    client: Any = None,
    rotation: prose.TeacherRotation | None = None,
    limiter: prose.TokenBucketLimiter | None = None,
    teachers: list[str] | None = None,
    budget_usd: float = DEFAULT_PAID_BUDGET_USD,
    spend: Any = None,
) -> dict[str, Any]:
    """Top-level `synth` orchestration. Returns a JSON-serializable report
    (also written to `<out_dir>/synth_report.json`) with teacher request/
    429 counts, verified/filler/failed counts per kind, and whether the
    run stopped early on a rate/budget limit.

    `type_mix`: per-type minimum planted-fact counts for this world (see
    `worlds.parse_type_mix`) -- `None` keeps the natural distribution.

    `client`/`rotation`/`limiter`: normally built fresh here (one client,
    one rotation, one rate-limit budget PER WORLD). A caller running many
    worlds in one process (`scripts/data/synth_batches.py`, "one shared
    rate limiter across worlds") passes its OWN already-built instances
    instead, so the free-tier budget (20 requests/minute, 1,000/day by
    default) is enforced across the WHOLE multi-world run, not reset to a
    fresh budget every world.

    `teachers` (added 2026-09-22, "paid teachers"): exact OpenRouter ids
    to rotate across, bypassing `discover_free_models` entirely -- used
    only when `rotation` is not already given (a caller that built its
    own `rotation` from `teachers` itself, e.g. `scripts/data/
    synth_batches.py` sharing one rotation across worlds, doesn't need
    this repeated). `prose.teachers_are_paid` decides free vs. paid
    (raises `prose.MixedTeacherListError` on a mixed list, never
    silently partitioned). When every id is paid: `requests_per_minute`/
    `requests_per_day` are overridden to 60/unlimited (no free-tier daily
    budget applies to a paid call) and, unless the caller already
    supplied one, `spend` defaults to the shared `var/data/silver/
    spend.json` meter.

    `budget_usd`/`spend`: only meaningful for a paid-teacher run --
    `spend` is `None` for every free run (the default, unchanged
    behavior). When `spend` is set (by this function's own `teachers`
    handling, or directly by the caller), every live call's real
    OpenRouter cost is recorded into it under `RunState.corpus_key`
    (`SYNTH_CORPUS_KEY`, "synthetic") and the run stops cleanly
    (`prose.PaidBudgetExceeded`, `state.stop_reason` containing `prose.
    PAID_BUDGET_STOP_MARKER`) the moment THIS corpus's own spend --
    `spend.corpus_usd(SYNTH_CORPUS_KEY)`, not the meter's shared total
    across every corpus -- reaches `budget_usd`, checked before every
    call, never mid-call."""
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = worlds.generate_world(
        seed=seed, num_accounts=num_accounts, per_kind=per_kind, type_mix=type_mix
    )
    registry = Registry(truth)

    truth_table_path = out_dir / "truth-table.yaml"
    with truth_table_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(truth, fh, sort_keys=False, allow_unicode=True)

    if rotation is None:
        if teachers is not None:
            paid = prose.teachers_are_paid(list(teachers))
            rotation = prose.TeacherRotation(list(teachers))
            if paid:
                requests_per_minute = PAID_TEACHER_RPM
                requests_per_day = PAID_TEACHER_RPD
                if spend is None:
                    spend = SpendMeter(path=default_spend_path(PROJECT_ROOT / "var"))
        else:
            api_key = prose._env_or_dotenv(PROJECT_ROOT, "OPENROUTER_API_KEY")
            all_free_models = prose.discover_free_models(api_key)
            if not all_free_models:
                raise RuntimeError(
                    "synth: no free (':free') OpenRouter models discovered -- check "
                    "OPENROUTER_API_KEY and network access; refusing to fall back to a paid model."
                )
            teacher_models = all_free_models[: prose.MAX_TEACHER_ROTATION_SIZE]
            rotation = prose.TeacherRotation(teacher_models)
    if limiter is None:
        limiter = prose.TokenBucketLimiter(
            requests_per_minute=requests_per_minute, requests_per_day=requests_per_day
        )
    if client is None:
        client = prose.build_openrouter_client(PROJECT_ROOT)
    progress = _synth_load_progress(out_dir)
    state = RunState()
    if spend is not None:
        state.spend = spend
        state.budget_usd = budget_usd

    # This world's own support-bot identity (build instructions "distinct
    # worlds"): never a fixed "bot-halvard-support"/"Halvard Support" --
    # see `worlds.py`'s docstring and `_synth_process_chat`.
    supplier_short = truth["meta"]["supplier"]["short"]
    bot_user_id = f"bot-{worlds._slug(supplier_short)}-support"
    bot_display_name = f"{supplier_short} Support"

    # `reduced=False`: the hand-built corpus's `reduced=True` noise counts
    # are hardcoded literals (2/15/20 -- `briefs.build_calls`/`build_
    # tickets`/`build_emails`), sized for THAT corpus's fixed noise
    # sections, not for `--per-kind`. `worlds.py` already sized `truth
    # ["sources"]["calls"/"tickets"/"emails"]["noise"]["count"]` to scale
    # with `per_kind` (build instructions item 4: "--per-kind 60 ...
    # producing calls, email, tickets" -- the count should track the flag).
    # `reduced=False` is the only way to make `build_calls` et al. read
    # that field instead of the literal. Call/noise-call LENGTH is still
    # clamped short below, independently of `reduced` -- see the comment
    # there for the rate-limit rationale (12k-25k-token calls at ~20
    # requests/minute would make even one batch take hours).
    rng = random.Random(seed)
    call_briefs = build_calls(truth, registry, rng, reduced=False)
    for cb in call_briefs:
        target = cb.extra.get("target_words") or 400
        cb.extra["target_words"] = min(target, 450 if cb.facts else 220)
    ticket_briefs = build_tickets(truth, registry, rng, reduced=False)
    email_briefs = build_emails(truth, registry, rng, reduced=False)
    slack_briefs = build_chat_briefs("slack", truth, registry)
    teams_briefs = build_chat_briefs("teams", truth, registry)
    crm_note_briefs = build_crm_notes(truth, registry)

    records: list[manifest_mod.SourceRecord] = []
    records += _synth_process_calls(
        call_briefs,
        out_dir,
        client=client,
        rotation=rotation,
        limiter=limiter,
        progress=progress,
        state=state,
    )
    if not state.stopped:
        records += _synth_process_tickets(
            ticket_briefs,
            registry,
            out_dir,
            client=client,
            rotation=rotation,
            limiter=limiter,
            progress=progress,
            state=state,
        )
    if not state.stopped:
        records += _synth_process_emails(
            email_briefs,
            registry,
            out_dir,
            client=client,
            rotation=rotation,
            limiter=limiter,
            progress=progress,
            state=state,
        )
    channel_to_account: dict[str, str] = {}
    if not state.stopped:
        slack_records, slack_channel_map = _synth_process_chat(
            "slack",
            slack_briefs,
            out_dir,
            client=client,
            rotation=rotation,
            limiter=limiter,
            progress=progress,
            state=state,
            bot_user_id=bot_user_id,
            bot_display_name=bot_display_name,
        )
        records += slack_records
        channel_to_account.update(slack_channel_map)
    if not state.stopped:
        teams_records, _ = _synth_process_chat(
            "teams",
            teams_briefs,
            out_dir,
            client=client,
            rotation=rotation,
            limiter=limiter,
            progress=progress,
            state=state,
            bot_user_id=bot_user_id,
            bot_display_name=bot_display_name,
        )
        records += teams_records
    if not state.stopped:
        records += _synth_process_crm_notes(
            crm_note_briefs,
            truth,
            registry,
            out_dir,
            client=client,
            rotation=rotation,
            limiter=limiter,
            progress=progress,
            state=state,
        )
    records += _synth_process_crm_structured(truth, registry, out_dir)

    # Slack channels.json / users.json (build instructions item 3's Slack
    # export shape). Every account that got at least one Slack source is a
    # channel; every contact + internal person + the synthetic support bot
    # is a user.
    channels_rows = [
        {"id": f"C{i + 1:03d}", "name": channel, "is_private": False, "account": account_id}
        for i, (channel, account_id) in enumerate(sorted(channel_to_account.items()))
    ]
    users_rows = (
        [
            {
                "id": c["id"],
                "name": c["email"].split("@")[0],
                "real_name": c["name"],
                "is_bot": False,
            }
            for c in truth["contacts"]
        ]
        + [
            {
                "id": p["id"],
                "name": p["email"].split("@")[0],
                "real_name": p["name"],
                "is_bot": False,
            }
            for p in truth["meta"]["internal_people"]
        ]
        + [
            {
                "id": bot_user_id,
                "name": f"{worlds._slug(supplier_short)}-support",
                "real_name": bot_display_name,
                "is_bot": True,
            }
        ]
    )
    structured.write_slack_channels(channels_rows, out_dir / "slack" / "channels.json")
    structured.write_slack_users(users_rows, out_dir / "slack" / "users.json")

    manifest = manifest_mod.build_manifest(
        mode="openrouter-free",
        reduced=True,
        generated_at=_dt.datetime.now(_dt.UTC).isoformat(),
        truth_table_path=truth_table_path,
        sources=records,
    )
    manifest_mod.write_manifest(manifest, out_dir / "manifest.json")
    problems = manifest_mod.validate(manifest, truth)

    by_kind: dict[str, int] = {}
    for s in manifest["sources"]:
        by_kind[s["type"]] = by_kind.get(s["type"], 0) + 1

    # Per-type breakdown (build instructions: batch-runner summaries need
    # "per-type counts"): `planted_by_type` is every non-distractor fact
    # the truth table intended to plant; `verified_by_type` is the subset
    # that actually landed with a verified marker (present in the
    # manifest's `facts` index -- see `manifest.build_manifest`).
    fact_type_by_id = {f["id"]: f["type"] for f in truth["facts"] if f["type"] != "distractor"}
    planted_by_type: dict[str, int] = {}
    for fact_type in fact_type_by_id.values():
        planted_by_type[fact_type] = planted_by_type.get(fact_type, 0) + 1
    verified_by_type: dict[str, int] = {}
    for fact_id in manifest["facts"]:
        fact_type = fact_type_by_id.get(fact_id)
        if fact_type:
            verified_by_type[fact_type] = verified_by_type.get(fact_type, 0) + 1

    report = {
        "seed": seed,
        "num_accounts": num_accounts,
        "per_kind": per_kind,
        "type_mix": type_mix or {},
        "supplier": truth["meta"]["supplier"]["name"],
        "teachers": rotation.teachers,
        "request_counts": rotation.request_counts,
        "rate_limit_counts": rotation.rate_limit_counts,
        "total_requests": rotation.total_requests,
        "total_rate_limits": rotation.total_rate_limits,
        "day_count": limiter.day_count,
        "verified": state.verified,
        "filler_used_after_failure": state.filler_used,
        "failed_sources": state.failures,
        "stopped_early": state.stopped,
        "stop_reason": state.stop_reason,
        "sources_by_kind": by_kind,
        "total_sources": len(manifest["sources"]),
        "planted_facts_placed": len(manifest["facts"]),
        "planted_by_type": planted_by_type,
        "verified_by_type": verified_by_type,
        "manifest_problems": problems,
    }
    (out_dir / "synth_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m corpus.generator.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate the corpus and manifest.")
    gen.add_argument("--out", default=str(PROJECT_ROOT / "corpus" / "out"))
    gen.add_argument("--mode", choices=["auto", "api", "manual", "openrouter"], default="auto")

    syn = sub.add_parser(
        "synth", help="Generate a fresh synthetic batch from a seed using free OpenRouter models."
    )
    syn.add_argument("--seed", type=int, required=True)
    syn.add_argument("--accounts", type=int, default=5)
    syn.add_argument("--per-kind", type=int, default=60)
    syn.add_argument("--out", required=True)
    syn.add_argument("--rpm", type=int, default=20, help="Free-tier requests-per-minute budget.")
    syn.add_argument("--rpd", type=int, default=1000, help="Free-tier requests-per-day budget.")
    syn.add_argument(
        "--type-mix",
        default=None,
        help=(
            "Per-type minimum planted-fact counts, e.g. "
            "'pricing=40,competitor_mention=40,risk=40,question=40,contact_change=30', "
            "or the literal 'balanced' for an equal floor share of every fact_type. "
            "Default: the natural FACT_TYPE_WEIGHTS distribution (no forcing)."
        ),
    )
    syn.add_argument(
        "--teachers",
        default=None,
        help=(
            "Comma-separated exact OpenRouter model ids, bypassing free-model discovery. "
            "All-paid (no id ends in ':free'): skips the free-tier daily budget, uses a "
            "60 requests/minute limiter, and stops at --budget-usd. All-free (every id "
            "ends in ':free'): behaves like the default discovery path, just with this "
            "exact list. Mixing free and paid ids is an error."
        ),
    )
    syn.add_argument(
        "--budget-usd",
        type=float,
        default=DEFAULT_PAID_BUDGET_USD,
        help="Paid-teacher dollar ceiling (only meaningful with an all-paid --teachers list).",
    )
    args = parser.parse_args(argv)

    if args.command == "generate":
        out_dir = Path(args.out)
        try:
            manifest, problems, cost_usd, usage_totals = generate(out_dir, args.mode)
        except prose.CorpusBudgetExceeded as exc:
            print(f"GENERATION STOPPED: {exc}")
            return 1
        by_type: dict[str, int] = {}
        for s in manifest["sources"]:
            by_type[s["type"]] = by_type.get(s["type"], 0) + 1
        total_tokens = sum(s["token_estimate"] for s in manifest["sources"])
        print(f"mode={manifest['mode']} reduced={manifest['reduced']}")
        print(f"sources by type: {by_type}")
        print(f"total sources: {len(manifest['sources'])}")
        print(f"total estimated tokens: {total_tokens}")
        print(f"planted facts placed: {len(manifest['facts'])}")
        if manifest["mode"] == "openrouter":
            print(
                f"openrouter: {usage_totals['calls']} calls, "
                f"{usage_totals['input_tokens']} input tokens, "
                f"{usage_totals['output_tokens']} output tokens, "
                f"${cost_usd:.4f} spent"
            )
        if problems:
            print(f"VALIDATION FAILED ({len(problems)} problems):")
            for p in problems:
                print(f"  - {p}")
            return 1
        print("VALIDATION PASSED")
        return 0

    if args.command == "synth":
        out_dir = Path(args.out)
        try:
            type_mix = worlds.parse_type_mix(args.type_mix, args.per_kind)
        except ValueError as exc:
            print(f"invalid --type-mix: {exc}")
            return 1
        teachers = (
            [t.strip() for t in args.teachers.split(",") if t.strip()] if args.teachers else None
        )
        if teachers is not None:
            try:
                prose.teachers_are_paid(teachers)
            except prose.MixedTeacherListError as exc:
                print(f"invalid --teachers: {exc}")
                return 1
        try:
            report = synth(
                seed=args.seed,
                num_accounts=args.accounts,
                per_kind=args.per_kind,
                out_dir=out_dir,
                requests_per_minute=args.rpm,
                requests_per_day=args.rpd,
                type_mix=type_mix,
                teachers=teachers,
                budget_usd=args.budget_usd,
            )
        except prose.PaidBudgetExceeded as exc:
            print(f"SYNTH STOPPED ({prose.PAID_BUDGET_STOP_MARKER}): {exc}")
            return 4
        except prose.DailyLimitExceeded as exc:
            print(f"SYNTH STOPPED (daily free-tier limit): {exc}")
            return 1
        print(
            f"seed={report['seed']} accounts={report['num_accounts']} per_kind={report['per_kind']}"
        )
        print(f"teachers: {report['teachers']}")
        print(f"request_counts: {report['request_counts']}")
        print(f"rate_limit_counts: {report['rate_limit_counts']}")
        print(f"sources by kind: {report['sources_by_kind']}")
        print(f"total sources: {report['total_sources']}")
        print(f"planted facts placed: {report['planted_facts_placed']}")
        print(
            f"verified: {report['verified']}, filler-after-failure: {report['filler_used_after_failure']}"
        )
        if report["stopped_early"]:
            print(f"STOPPED EARLY: {report['stop_reason']}")
            if prose.PAID_BUDGET_STOP_MARKER in report["stop_reason"]:
                return 4
        if report["failed_sources"]:
            print(f"FAILED SOURCES ({len(report['failed_sources'])}):")
            for f in report["failed_sources"]:
                print(f"  - {f['source_id']}: {f['problems']}")
        if report["manifest_problems"]:
            print(f"VALIDATION FAILED ({len(report['manifest_problems'])} problems):")
            for p in report["manifest_problems"]:
                print(f"  - {p}")
            return 1
        print("VALIDATION PASSED")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
