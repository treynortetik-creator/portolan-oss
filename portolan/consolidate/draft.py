"""`draft_synthesis` and `check` (`01-spec/quality-loop.md` "The
consolidation pass": "a cheap model rewrites the synthesis ... from the
claim list. It is told: every claim id must appear, every citation marker
must survive verbatim, no new claims" / "Check, mechanically: compare the
set of citation markers and claim ids before and after. Any missing means
reject").

**Deviation from that spec text, review finding G7 (fix wave
2026-09-12).** "Every claim id must appear" is what broke archiving:
`consolidate.archive.select_claims_to_archive` skips any claim the
*current* synthesis still cites, oldest-first among the rest -- exactly
right, per its own docstring, quoting the wiki-schema growth-guard warning
that evicting by age alone removes the most-cited facts. But if the
synthesis is forced to cite EVERY remaining claim, every single one
becomes "load-bearing" the moment it's consolidated once -- so the next
pass can only ever archive claims added *since* (the only ones not yet
cited), never the genuinely old ones the growth threshold exists to clear
out. A real customer graph would see archiving invert itself after
exactly one pass and never fire correctly again. The fix: the drafted
synthesis names the five to ten claims that matter most for a reader right
now, not the whole list -- "no invented citation/id" is still a hard
mechanical rule, "cites everything" is not.

**Claim-id tags, a convention this module adds.** A rendered claim line
(`portolan.notes.render._render_claim_line`) already ends in a citation
marker (`^[source-slug @ locator]`) but carries no visible token for the
claim's own database id -- there was never a reason to show one to a
reader before now. Mechanically checking "every claim id must appear" (as
distinct from "every citation marker must survive") needs *some* textual
token per claim id, so this module defines one, `[claim:<id>]`, and
`draft_synthesis`'s prompt requires it appear immediately after each
citation marker it uses. This is scoped entirely to a note's freeform
`## Synthesis` prose (`note.synthesis`) -- it never touches the append-
only `## Claims` section's own rendering (`portolan.notes.render`, not
edited by this wave) -- and is worth folding into that renderer directly
in a later wave if claim ids turn out to be useful to show elsewhere too
(flagged in this wave's report).
"""

from __future__ import annotations

import re

from portolan.consolidate.models import CheckResult
from portolan.lint.checks import _CODE_RE, _QUOTE_RE, _RELATIVE_TIME_RE, _mask
from portolan.notes.models import Claim, NoteRow
from portolan.notes.render import CITATION_RE

__all__ = ["CLAIM_ID_TAG_RE", "DRAFT_SYSTEM", "MAX_SYNTHESIS_LINES", "check", "draft_synthesis"]

MAX_SYNTHESIS_LINES = 40

# The floor `check()` enforces (module docstring, G7): "the five to ten
# claims that matter", read as "at least five" -- or every claim, for a
# note with fewer than five remaining (nothing to trim down to).
MIN_SYNTHESIS_CITATIONS = 5

CLAIM_ID_TAG_RE = re.compile(r"\[claim:(\d+)\]")

DRAFT_SYSTEM = (
    "You write the 'Synthesis' section of a customer-intelligence note: a "
    "short, current-state prose summary built from the claim list you are "
    "given below. Every sentence must be grounded in one of those claims. "
    "Rules, checked mechanically after you answer, so follow them "
    "exactly:\n\n"
    "1. Every claim you use must carry its citation marker verbatim, "
    "exactly as given: ^[source-slug @ locator].\n"
    "2. Immediately after that citation marker, append the claim's id "
    "verbatim in the form [claim:<id>] (also exactly as given).\n"
    "3. Cite the five to ten claims below that matter most for a reader "
    "trying to understand this account or entity right now -- you do not "
    "need to cite every claim in the list, and should not pad the "
    "synthesis just to do so. Never invent a citation marker, claim id, "
    "claim, fact, or entity not in the list.\n"
    "4. Durable, absolute language only -- no 'recently', 'currently', "
    "'this month', or any other relative-time phrase.\n"
    "5. Keep the whole synthesis under 40 lines.\n"
    "6. Each claim line carries the date it was true or recorded. When two "
    "claims disagree, the one with the later date is the current state; say "
    "what changed and when, with the dates written out.\n\n"
    "Reply with the synthesis prose only, nothing else."
)


def _claim_line(claim: Claim) -> str:
    marker = f"^[{claim.source_slug} @ {claim.locator}]"
    # The world date when the claim has one, else the day it was recorded:
    # without it the model cannot order events or write absolute dates.
    dated = claim.valid_from or claim.recorded.date()
    return f"- claim {claim.id} [{claim.fact_type}] ({dated:%Y-%m-%d}): {claim.claim_text} {marker}"


def draft_synthesis(llm, note: NoteRow, claims: list[Claim]) -> str:
    """Ask `llm` (the cheap model on the first attempt, the strong model
    on retry -- the caller decides which client to pass) to rewrite
    `note`'s synthesis from `claims`. `surface="consolidate"`."""
    claims_block = "\n".join(_claim_line(c) for c in claims)
    user = (
        f"Note: {note.title} ({note.type.value})\n\n"
        f"Claims:\n{claims_block}\n\n"
        "Write the synthesis now."
    )
    result = llm.complete(system=DRAFT_SYSTEM, user=user, surface="consolidate")
    return (result.text or "").strip()


def check(draft: str, claims: list[Claim]) -> CheckResult:
    """Mechanical verification of a drafted synthesis against `claims`
    (module docstring, review finding G7): every citation marker and
    claim-id tag the draft uses must be real (no invented ones -- still a
    hard failure), and it must cite at least `MIN_SYNTHESIS_CITATIONS`
    claims (or all of them, if fewer remain) -- never that it cites
    *every* claim given."""
    expected_citations = {f"{c.source_slug} @ {c.locator}" for c in claims}
    found_citations = {
        f"{m.group(1).strip()} @ {m.group(2).strip()}" for m in CITATION_RE.finditer(draft)
    }
    unknown_citations = sorted(found_citations - expected_citations)

    expected_ids = {c.id for c in claims if c.id is not None}
    found_ids = {int(m.group(1)) for m in CLAIM_ID_TAG_RE.finditer(draft)}
    unknown_ids = sorted(found_ids - expected_ids)
    cited_ids = sorted(found_ids & expected_ids)

    min_required = min(MIN_SYNTHESIS_CITATIONS, len(expected_ids))
    too_few = len(cited_ids) < min_required

    non_blank_lines = [line for line in draft.splitlines() if line.strip()]
    line_count = len(non_blank_lines)
    too_long = line_count > MAX_SYNTHESIS_LINES

    # Same masking `portolan.lint.checks.check_relative_time` applies
    # before running the relative-time regex: a fenced/inline code span or
    # a quoted customer sentence is not the note itself making an undated
    # claim (`portolan.lint.checks` module docstring).
    masked = _mask(_mask(draft, _CODE_RE), _QUOTE_RE)
    relative_hits = sorted({m.group(0).lower() for m in _RELATIVE_TIME_RE.finditer(masked)})

    reasons: list[str] = []
    if unknown_citations:
        reasons.append(f"unknown citation marker(s): {', '.join(unknown_citations)}")
    if unknown_ids:
        reasons.append(f"unknown claim id tag(s): {unknown_ids}")
    if too_few:
        reasons.append(
            f"cites only {len(cited_ids)} claim(s), fewer than the required {min_required}"
        )
    if too_long:
        reasons.append(f"{line_count} lines (limit {MAX_SYNTHESIS_LINES})")
    if relative_hits:
        reasons.append("relative-time language: " + ", ".join(relative_hits))

    return CheckResult(
        ok=not reasons,
        line_count=line_count,
        too_long=too_long,
        cited_claim_ids=cited_ids,
        min_required_citations=min_required,
        too_few_citations=too_few,
        unknown_citations=unknown_citations,
        unknown_claim_ids=unknown_ids,
        relative_time_hits=relative_hits,
        reasons=reasons,
    )
