"""The weekly report assembler (`01-spec/surfacing-and-weekly-report.md`
"The report is a node", `01-spec/portolan-spec-v2.md` sec 7).

`build_weekly` runs every feed in this package for one seven-day window,
writes the result as a `report` note (`report-<ISO week>`), upserts a
`reports` row, and returns the `ReportResult`. `render_email` renders
that result as markdown: the note is the truth, the email a copy.

CLI: `python -m portolan.report weekly (--as-of YYYY-MM-DD | --week
YYYY-Www) [--llm PROVIDER] [--prose-model MODEL_ID] [--embed
fake|fastembed] [--corpus PATH] [--out DIR] [--clock record|world]
[--research] [--propose-done]`, plus the `research` and `propose-done`
subcommands. `--llm` overrides `LLM_PROVIDER` for one run
(`_build_llm`); `--corpus` names the CRM export ARR and region come from.

`--clock record` (default) keys the diff's created/updated split on
`notes.created`/`updated` (write time); `world` keys it on
`claims.recorded` (the source's own date), so a corpus backfilled in one
sitting still reports a past week correctly. Themes, standing questions
and the actions feed follow the clock too. Coverage and "resolved this
window" are write-time only; under `world` the report says so in its
notes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    # Types only; the researcher modules are imported inside the functions.
    from portolan.report.researcher import CandidateResearchOutcome, Researcher
    from portolan.report.support_judge import SupportJudgeFn

from portolan.llm.base import LLMClient
from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
from portolan.report import diff as diff_feed
from portolan.report import executive as executive_feed
from portolan.report import outliers as outliers_feed
from portolan.report import public_signals as public_signals_feed
from portolan.report import questions as questions_feed
from portolan.report import synthesis as synthesis_feed
from portolan.report import themes as themes_feed
from portolan.report import waves as waves_feed
from portolan.report.accounts import (
    account_arr_by_note_slug,
    account_region_by_note_slug,
    load_account_arr_via_connector,
    load_account_regions_via_connector,
)
from portolan.report.models import (
    ActionRow,
    ActionsFeed,
    CoverageSummary,
    ExecutiveSummary,
    OutlierClaim,
    PublicSignal,
    ReportResult,
    SignalDetail,
    ThemePriority,
    ThemeReport,
)
from portolan.report.numbers import NumberFacts, check_numerals, queue_numeral_rejection
from portolan.report.priorities import compute_theme_priorities, format_priority_line
from portolan.report.resolved_language import RESOLVED_LANGUAGE_RE, resolved_language_rejected
from portolan.report.util import (
    citation_markers_in,
    iso_week_str,
    report_note_slug,
    validate_citations,
    week_bounds,
)
from portolan.search.embed import Embedder
from portolan.util import clock as clock_module

__all__ = [
    "UnknownCandidateSlugError",
    "build_weekly",
    "contact_changes_feed",
    "main",
    "render_email",
    "run_research_harness",
]

HEADLINE_SYSTEM = (
    "You write the headline for a weekly customer-signals report: three to "
    "five sentences of plain narrative prose, from four mechanical feeds "
    "given to you as compact JSON: the diff (what's new, updated, "
    "superseded, gone quiet), themes by movement, standing-question "
    "answers, and outliers. The report's own 'Themes by movement' section "
    "already lists every theme's own signal count, account count and ARR "
    "mechanically -- your job is NOT to restate that table one theme per "
    "sentence ('Theme X is rising with $Y ARR across accounts A, B. Theme "
    "Z is new with $W ARR across accounts C, D.'). Instead, tell a reader "
    "what actually changed this week and why it matters: pick out the "
    "most significant story or two (by ARR, by account count, or because "
    "it just changed state -- newly resolved, newly rising, a reversal), "
    "say what is driving it, and connect it to what a reader should "
    "watch or do next. Write it as connected prose a person would "
    "actually say out loud, not a list of formulaic fact-sentences. Do "
    "not invent anything not present in the JSON. Every number you state "
    "must come from the JSON's own 'figures' table or appear verbatim "
    "elsewhere in the JSON -- never compute, round, sum, or estimate a "
    "number yourself; a number is welcome in a sentence when it is part "
    "of the real story, not required in every sentence. Cite claims with "
    "their exact marker where the JSON gives one -- reproduce it exactly "
    "as given, character for character, INCLUDING the leading '^' -- "
    "'^[source @ locator]', never 'source @ locator' or '[source @ "
    "locator]' without the caret. Write account and "
    "theme names as plain words only -- never wrap one in double square "
    "brackets like '[[slug]]'; the report's own mechanical sections "
    "already link every account and theme by name, so your sentences "
    "need only say the name once, in prose. Absolute dates only, "
    "never 'this week' or 'recently'. Never say a theme is resolved, or "
    "use the word 'resolution', for a theme whose own 'movement' value "
    "is not exactly 'resolved' -- 'resolution proposed' means a judge "
    "proposed it and a human has not yet confirmed it, which is not the "
    "same thing as resolved. Reply as JSON with a 'sentences' list."
)


class _Headline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentences: list[str] = Field(default_factory=list)


def _headline_json_schema() -> dict:
    return _Headline.model_json_schema()


# The vocabulary and the rule live in `portolan.report.resolved_language`.
_RESOLVED_LANGUAGE_RE = RESOLVED_LANGUAGE_RE

# Models write `[[slug]]` despite the prompt, and it renders as a repeat of
# the name the prose already says; only the token is removed, not the sentence.
_STRAY_WIKILINK_RE = re.compile(r"\(\[\[[a-z0-9_-]+\]\]\)|\[\[[a-z0-9_-]+\]\]")


def _strip_stray_wikilinks(sentence: str) -> str:
    stripped = _STRAY_WIKILINK_RE.sub("", sentence)
    stripped = re.sub(r"\s+([.,;:!?])", r"\1", stripped)  # no space before punctuation
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


# A `[... @ ...]` without the leading `^` is a citation with the caret
# dropped: `validate_citations` sees no marker in it, so the sentence is
# dropped as a bad citation rather than patched.
_MALFORMED_CITATION_RE = re.compile(r"(?<!\^)\[[^\]@]+@[^\]]*\]")


def _queue_resolved_language_rejection(conn, *, sentence: str) -> None:
    """Queue a headline sentence that `resolved_language_rejected` refused
    (`review_queue`, `kind='other'`); the sentence is dropped, never
    trimmed. The `reason` literal is older than the per-sentence rule and
    tests assert it."""
    payload = {
        "reason": "resolved_language_without_a_resolved_theme",
        "surface": "headline",
        "note_slug": None,
        "detail": sentence,
    }
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('other', %s::jsonb)",
            (json.dumps(payload),),
        )
    conn.commit()


def _actions_feed(conn, *, since: date, as_of: date, clock: str = "record") -> ActionsFeed:
    """Open, overdue and resolved-this-window actions as of `as_of`. An
    action created after `as_of` is left out of every list; creation is
    `notes.created` under `clock="record"`, or under `clock="world"` the
    `recorded` date of the note's earliest claim (not `valid_from`, which
    holds the due date). A note with no claims -- an action confirmed
    from a researched card has none -- falls back to `notes.created`.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.id, a.note_slug, a.what, re.canonical_name, re.crm_id, a.due, a.status,
                   n.created, cl.recorded
            FROM actions a
            LEFT JOIN registry_entities re ON re.id = a.owner_entity_id
            JOIN notes n ON n.slug = a.note_slug
            LEFT JOIN LATERAL (
                SELECT c.recorded FROM claims c
                WHERE c.note_slug = a.note_slug ORDER BY c.recorded LIMIT 1
            ) cl ON true
            ORDER BY a.id
            """
        )
        rows = cur.fetchall()

    open_rows: list[ActionRow] = []
    overdue_rows: list[ActionRow] = []
    resolved_rows: list[ActionRow] = []

    with conn.cursor() as cur:
        for (
            aid,
            note_slug,
            what,
            owner,
            owner_crm_id,
            due,
            status,
            note_created,
            recorded,
        ) in rows:
            if clock == "world" and recorded is not None:
                creation_date = clock_module.world_date(recorded)
            else:
                creation_date = clock_module.world_date(note_created)
            if creation_date > as_of:
                continue

            row = ActionRow(
                id=aid,
                note_slug=note_slug,
                what=what,
                owner=owner,
                owner_crm_id=owner_crm_id,
                due=due,
                status=status,
            )
            if status == "open":
                open_rows.append(row)
                if due is not None and due < as_of:
                    overdue_rows.append(row)
            elif status == "overdue":
                overdue_rows.append(row)
                open_rows.append(row)
            elif status in ("done", "dropped"):
                # When it closed: the latest `edit_records` row setting done
                # or dropped (`actions` has no such column), else `notes.updated`.
                cur.execute(
                    "SELECT edited_at FROM edit_records WHERE note_slug = %s "
                    "AND after_text IN ('done', 'dropped') ORDER BY edited_at DESC LIMIT 1",
                    (note_slug,),
                )
                edit_row = cur.fetchone()
                if edit_row is not None:
                    closed_at = edit_row[0]
                else:
                    cur.execute("SELECT updated FROM notes WHERE slug = %s", (note_slug,))
                    updated_row = cur.fetchone()
                    closed_at = updated_row[0] if updated_row else None
                # `world_date`, not `.date()`: the deployment's zone, not the session's.
                if closed_at is not None and since <= clock_module.world_date(closed_at) <= as_of:
                    resolved_rows.append(row)

    return ActionsFeed(open=open_rows, overdue=overdue_rows, resolved_this_week=resolved_rows)


def _sample_miss_count(conn, *, since: date, as_of: date) -> int:
    """Sum of `facts_found` across this window's `filter_audit` changelog
    rows (`portolan.pipeline._run_filter_audit`): facts the extractor
    found in sampled sources the filter had excluded."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COALESCE(SUM((details->>'facts_found')::int), 0)
            FROM changelog
            WHERE kind = 'filter_audit'
              AND {clock_module.world_date_sql("occurred_at")} BETWEEN %s AND %s
              AND details ? 'facts_found'
            """,
            (since, as_of),
        )
        (total,) = cur.fetchone()
    return total


def _coverage_summary(conn, *, since: date, as_of: date) -> CoverageSummary:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT seen, extracted, partial, failed, sampled, filtered
            FROM coverage_ledger
            WHERE day BETWEEN %s AND %s
            """,
            (since, as_of),
        )
        rows = cur.fetchall()

    summary = CoverageSummary(since=since, as_of=as_of)
    for seen, extracted, partial, failed, sampled, filtered in rows:
        summary.seen += seen
        summary.extracted += extracted
        summary.partial += partial
        summary.failed += failed
        summary.sampled += sampled
        for reason, count in (filtered or {}).items():
            summary.filtered_by_reason[reason] = summary.filtered_by_reason.get(reason, 0) + count
    summary.sample_miss_count = _sample_miss_count(conn, since=since, as_of=as_of)
    return summary


def contact_changes_feed(conn, *, since: date, as_of: date) -> list[OutlierClaim]:
    from portolan.report.accounts import home_account_slug

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, note_slug, fact_type, claim_text, source_slug, locator, recorded
            FROM claims
            WHERE fact_type = 'contact_change'
              AND {clock_module.world_date_sql("recorded")} BETWEEN %s AND %s
            ORDER BY recorded, id
            """,
            (since, as_of),
        )
        rows = cur.fetchall()
    result = []
    for claim_id, note_slug, fact_type, claim_text, source_slug, locator, recorded in rows:
        account = home_account_slug(
            conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
        )
        result.append(
            OutlierClaim(
                claim_id=claim_id,
                note_slug=note_slug,
                account_slug=account,
                fact_type=fact_type,
                claim_text=claim_text,
                source_slug=source_slug,
                locator=locator,
                recorded=clock_module.world_date(recorded),
            )
        )
    return result


def _headline(
    conn,
    llm: LLMClient,
    diff,
    themes,
    questions,
    outliers,
    *,
    prose_model: str | None = None,
    failure_reasons: list[str] | None = None,
) -> list[str]:
    """The headline sentences, each gated on its own: a stray `[[slug]]`
    is stripped, then a sentence is dropped for a citation marker not in
    the payload (or one missing its caret), a numeral outside the
    `NumberFacts` table (queued `numeral_not_supplied`), or resolved
    language about an unresolved theme. Superseded-fact and outlier
    numbers are licensed only with their own citation marker.

    `failure_reasons`, when a list, gets one plain-language reason
    whenever this returns `[]`: the model call raised, returned nothing,
    or every sentence was dropped. `prose_model` is the model for the one
    `llm.complete()` call; `None` keeps `llm`'s default.
    """
    superseded_lines = [
        f"{f.new_text} ^[{f.new_source_slug} @ {f.new_locator}]" for f in diff.facts_superseded
    ]
    outlier_lines = [f"{o.claim_text} ^[{o.source_slug} @ {o.locator}]" for o in outliers]

    facts = NumberFacts()
    facts.add("notes_created", len(diff.notes_created))
    facts.add("notes_updated", len(diff.notes_updated))
    facts.add("quiet_30_count", len(diff.quiet_accounts.quiet_30))
    facts.add("facts_superseded_count", len(superseded_lines))
    facts.add("outliers_count", len(outlier_lines))
    # Thresholds and the window length are facts too, so "quiet for 30
    # days" passes the numeral gate.
    for threshold_days in diff_feed.QUIET_ACCOUNT_THRESHOLDS:
        facts.add(f"quiet_threshold_{threshold_days}_days", threshold_days)
    facts.add("window_days", (diff.as_of - diff.since).days + 1)
    facts.add("theme_resolved_after_weeks", themes_feed.THEME_RESOLVED_AFTER_WEEKS)
    for t in themes.themes:
        facts.add(f"{t.slug}:signal_count", t.signal_count)
        facts.add(f"{t.slug}:linked_arr", t.linked_arr, f"{t.linked_arr:,.0f}")
    for f, line in zip(diff.facts_superseded, superseded_lines, strict=True):
        facts.harvest(line, marker=(f.new_source_slug, f.new_locator))
    for o, line in zip(outliers, outlier_lines, strict=True):
        facts.harvest(line, marker=(o.source_slug, o.locator))

    payload = {
        "diff": {
            "notes_created": len(diff.notes_created),
            "notes_updated": len(diff.notes_updated),
            "facts_superseded": superseded_lines,
            "quiet_30": diff.quiet_accounts.quiet_30,
        },
        "themes": [
            {
                "name": t.name,
                # Spelled out: models read "resolution proposed" as resolved.
                "movement": (
                    "resolution proposed (awaiting human confirmation -- NOT resolved)"
                    if t.movement == "resolution proposed"
                    else t.movement
                ),
                "signal_count": t.signal_count,
                "accounts": t.account_slugs,
                "linked_arr": t.linked_arr,
            }
            for t in themes.themes
        ],
        "standing_questions": [{"question": q.question, "answer": q.answer_md} for q in questions],
        "outliers": outlier_lines,
    }
    payload_text = json.dumps(payload, default=str)
    # Themes and standing-question answers have no single claim to bind a
    # number to, so their numbers are licensed anywhere.
    facts.harvest(json.dumps(payload["themes"], default=str))
    facts.harvest(json.dumps(payload["standing_questions"], default=str))
    payload["figures"] = {name: rendering for name, (_v, rendering) in facts.items()}

    try:
        result = llm.complete(
            system=HEADLINE_SYSTEM,
            user=json.dumps(payload, default=str),
            json_schema=_headline_json_schema(),
            model=prose_model,
            surface="report",
        )
    except Exception as exc:  # noqa: BLE001 -- a model failure must not fail the whole report
        if failure_reasons is not None:
            failure_reasons.append(f"the model call failed ({type(exc).__name__}: {exc})")
        return []
    parsed = _Headline.model_validate(result.data)

    allowed_markers = citation_markers_in(payload_text)
    # A quoted span is the customer's own words only when it appears in
    # a claim-derived line the model was handed (`resolved_language`).
    grounded_texts = [*outlier_lines, *superseded_lines]

    kept: list[str] = []
    bad_citation = 0
    bad_numeral = 0
    bad_resolved_language = 0
    for raw_sentence in parsed.sentences:
        # Strip first, so no check runs on text that is about to be removed.
        sentence = _strip_stray_wikilinks(raw_sentence)
        if validate_citations(sentence, allowed_markers) is not None:
            bad_citation += 1
            continue
        if _MALFORMED_CITATION_RE.search(sentence):
            bad_citation += 1
            continue
        numeral_violations = check_numerals(sentence, facts)
        if numeral_violations:
            queue_numeral_rejection(
                conn,
                note_slug=None,
                surface="headline",
                violations=numeral_violations,
                text=sentence,
                facts=facts,
            )
            bad_numeral += 1
            continue
        if resolved_language_rejected(sentence, themes.themes, grounded_texts=grounded_texts):
            _queue_resolved_language_rejection(conn, sentence=sentence)
            bad_resolved_language += 1
            continue
        kept.append(sentence)

    if not kept and failure_reasons is not None:
        if not parsed.sentences:
            failure_reasons.append("the model returned no sentences")
        else:
            parts = []
            if bad_citation:
                parts.append(f"{bad_citation} with an unsupported citation")
            if bad_numeral:
                parts.append(f"{bad_numeral} with an unsupplied numeral")
            if bad_resolved_language:
                parts.append(f"{bad_resolved_language} misusing resolved-language")
            failure_reasons.append(f"every headline sentence was rejected ({', '.join(parts)})")
    return kept


def _signals_fold_block(signals: list[SignalDetail], *, chip_label: str, indent: str = "  ") -> str:
    """Markdown for the report's expandable "N signals" fold. Signals are
    verbatim claim data, not model output, so no citation/numeral gate.

    The block sits between `<!--signals:LABEL-->` and `<!--/signals-->`
    sentinel lines, not raw HTML (the renderer keeps `html: False`):
    `portolan.web.markdown` turns the pair into a `<details>` fold and the
    .md download strips the sentinels, leaving the bullet list
    (`tests/test_web.py`). Returns `""` for no signals.
    """
    if not signals:
        return ""
    lines = [f"<!--signals:{chip_label}-->"]
    for sig in signals:
        acct = f"[[{sig.account_slug}]]" if sig.account_slug else "(no account)"
        label = sig.fact_type.replace("_", " ")
        lines.append(
            f"{indent}- **{label}** — {acct}: {sig.claim_text} "
            f"^[{sig.source_slug} @ {sig.locator}] · [[{sig.note_slug}]]"
        )
        # The quote often repeats `claim_text`; show it only when it adds something.
        if sig.quote and not _quote_adds_nothing(sig.claim_text, sig.quote):
            lines.append(f'{indent}  - "{sig.quote}"')
    lines.append("<!--/signals-->")
    return "\n".join(lines)


_NORMALIZE_FOR_COMPARISON_RE = re.compile(r"[^a-z0-9]+")


def _normalize_for_comparison(text: str) -> str:
    return _NORMALIZE_FOR_COMPARISON_RE.sub(" ", text.lower()).strip()


def _quote_adds_nothing(claim_text: str, quote: str) -> bool:
    """True when `quote` equals `claim_text` or contains it, compared
    lowercased with every run of non-alphanumerics collapsed to a space
    (so quote marks and punctuation never count as a difference)."""
    norm_claim = _normalize_for_comparison(claim_text)
    norm_quote = _normalize_for_comparison(quote)
    if not norm_claim or not norm_quote:
        return False
    return norm_claim == norm_quote or norm_claim in norm_quote


def _render_body(
    week: str,
    headline: list[str],
    diff,
    themes,
    waves,
    questions,
    outliers,
    actions: ActionsFeed,
    coverage: CoverageSummary,
    contact_changes: list[OutlierClaim],
    clock: str = "record",
    headline_reason: str | None = None,
    executive_summary: ExecutiveSummary | None = None,
    priorities: list[ThemePriority] | None = None,
    action_card_lines: list[str] | None = None,
    public_signals: list[PublicSignal] | None = None,
) -> str:
    """The report body markdown. The "### Executive summary" section is the
    only narrative and renders only when `executive_summary` is given:
    `priorities` lines (`format_priority_line`) open "What needs
    attention" ahead of the summary's own `attention` bullets, and
    `action_card_lines` (`researcher.action_card_report_lines`) open
    "Suggested actions" ahead of its free-text bullets. `headline` and
    `headline_reason` are accepted but not rendered. "### Public signals
    this week" renders only when `public_signals` is not `None`. Internal
    caveats (world-clock coverage, a failed summary) go in a closing
    "### Notes" list, not inline."""
    footnotes: list[str] = []
    # Coverage and "resolved this window" have only write-time dates, so a
    # world-clock backfill of a past week reads zero; say why.
    if clock == "world":
        footnotes.append(
            "Coverage figures are record time only -- this report's window is "
            "world-clock; they reflect real ingest/write time, not the source's own date."
        )
        footnotes.append(
            'Actions\' "Resolved this window" count is record time only -- this '
            "report's window is world-clock; it reflects real ingest/write time, "
            "not the source's own date."
        )
    if executive_summary and executive_summary.failure_reason:
        footnotes.append(f"executive summary unavailable: {executive_summary.failure_reason}")

    parts: list[str] = []

    if executive_summary and (
        executive_summary.this_week
        or executive_summary.attention
        or executive_summary.by_account
        or executive_summary.suggested_actions
        or priorities
        or action_card_lines
    ):
        parts.append("### Executive summary")
        if executive_summary.this_week:
            parts.append("**This week.** " + " ".join(executive_summary.this_week))
            parts.append("")
        if priorities or executive_summary.attention:
            parts.append("**What needs attention**")
            # Code-rendered ranked rows first, then the model's bullets.
            parts.extend(f"- {format_priority_line(p)}" for p in (priorities or []))
            parts.extend(f"- {s}" for s in executive_summary.attention)
            parts.append("")
        if executive_summary.by_account:
            parts.append("**By account**")
            parts.append("")
            for entry in executive_summary.by_account:
                # Blank lines around both, or CommonMark merges them into one paragraph.
                parts.append(f"**[[{entry.account_slug}]]**")
                parts.append("")
                parts.append(entry.paragraph)
                parts.append("")
        if executive_summary.suggested_actions or action_card_lines:
            parts.append("**Suggested actions**")
            # Queued action cards first, then the model's bullets.
            parts.extend(f"- {s}" for s in (action_card_lines or []))
            parts.extend(f"- {s}" for s in executive_summary.suggested_actions)
            parts.append("")

    parts.append("### Themes by movement")
    if themes.themes:
        by_movement: dict[str, list] = {}
        for t in themes.themes:
            by_movement.setdefault(t.movement, []).append(t)
        # Reading order. Must list every `Movement` value, or those themes
        # are skipped; "resolution proposed" awaits a human.
        for movement in ("new", "rising", "falling", "resolved", "resolution proposed"):
            for t in by_movement.get(movement, []):
                # `linked_arr` counts top-level accounts only, so the count
                # and list must too; rolled-up children are named after.
                top_level_slugs = sorted(set(t.account_slugs) - set(t.rolled_up))
                accounts_txt = ", ".join(f"[[{a}]]" for a in top_level_slugs)
                arr_note = ""
                if t.rolled_up:
                    rollup_txt = ", ".join(
                        f"{child} under {parent}" for child, parent in sorted(t.rolled_up.items())
                    )
                    arr_note = f" (rolled up: {rollup_txt})"
                if t.arr_unknown_accounts:
                    plural = "account" if t.arr_unknown_accounts == 1 else "accounts"
                    arr_note += f" ({t.arr_unknown_accounts} {plural} excluded, ARR unknown)"
                # A peak is an annotation on a past week, not a movement.
                peak_note = ""
                if t.peaked_week:
                    plural = "signal" if t.peaked_count == 1 else "signals"
                    peak_note = (
                        f" (peaked week of {t.peaked_week:%Y-%m-%d}, {t.peaked_count} new {plural})"
                    )
                # The wikilink renders as the theme's title; no quoted name after it.
                signals_word = "signal" if t.signal_count == 1 else "signals"
                accounts_word = "account" if len(top_level_slugs) == 1 else "accounts"
                parts.append(
                    f"- **{movement}** — [[{t.slug}]]: {t.signal_count} {signals_word} across "
                    f"{len(top_level_slugs)} {accounts_word} ({accounts_txt}), "
                    f"${t.linked_arr:,.0f} ARR{arr_note}{peak_note}"
                )
                # Every signal behind the count, folded.
                plural = "signal" if t.signal_count == 1 else "signals"
                fold = _signals_fold_block(t.all_signals, chip_label=f"{t.signal_count} {plural}")
                if fold:
                    parts.append(fold)
                # Open resolution proposals: only the cited evidence quote,
                # never the judge's free-text reason (no numeral check).
                for pending in t.pending_resolutions:
                    parts.append(
                        f'  - Resolutions proposed: "{pending.evidence}" '
                        f"^[{pending.source_slug} @ {pending.locator}]"
                    )
    else:
        parts.append("- no themes touched this window")
    parts.append("")

    # Waves (`portolan.report.waves`): mechanical, no model call.
    parts.append("### Waves")
    if waves.waves:
        for w in waves.waves:
            # Top-level accounts only: `_top_level_accounts` puts each leaf
            # in exactly one of top-level or `rolled_up`.
            top_level_count = len(w.account_slugs) - len(w.rolled_up)
            top_level_slugs = sorted(set(w.account_slugs) - set(w.rolled_up))
            accounts_txt = ", ".join(f"[[{a}]]" for a in top_level_slugs)
            arr_note = ""
            if w.rolled_up:
                rollup_txt = ", ".join(
                    f"{child} under {parent}" for child, parent in sorted(w.rolled_up.items())
                )
                arr_note = f" (rolled up: {rollup_txt})"
            if w.unknown_arr_count:
                plural = "account" if w.unknown_arr_count == 1 else "accounts"
                arr_note += f" ({w.unknown_arr_count} {plural} excluded, ARR unknown)"
            parts.append(
                f"- week of {w.week_of:%Y-%m-%d} — **{w.fact_type}**, {w.product_family}: "
                f"{top_level_count} accounts ({accounts_txt}), "
                f"${w.linked_arr:,.0f} ARR{arr_note}"
            )
            # Every claim in the wave, folded.
            claim_plural = "claim" if len(w.claim_ids) == 1 else "claims"
            wave_fold = _signals_fold_block(
                w.all_signals, chip_label=f"{len(w.claim_ids)} {claim_plural}"
            )
            if wave_fold:
                parts.append(wave_fold)
    else:
        parts.append("- none this window")
    parts.append("")

    parts.append("### Standing questions")
    if questions:
        for q in questions:
            parts.append(f"- **{q.question}** ({q.reader}): {q.answer_md}")
    else:
        parts.append("- no active standing questions")
    parts.append("")

    parts.append("### Outliers")
    if outliers:
        for o in outliers:
            acct = f"[[{o.account_slug}]]" if o.account_slug else "(no account)"
            parts.append(f"- {acct}: {o.claim_text} ^[{o.source_slug} @ {o.locator}]")
    else:
        parts.append("- none this window")
    parts.append("")

    if public_signals is not None:
        parts.extend(public_signals_feed.render_public_signals(public_signals))
        if parts[-1] != "":
            parts.append("")

    parts.append("### Actions")
    parts.append(f"- Open: {len(actions.open)}")
    parts.append(f"- Overdue: {len(actions.overdue)}")
    parts.append(f"- Resolved this window: {len(actions.resolved_this_week)}")
    for a in actions.open:
        due_txt = f", due {a.due:%Y-%m-%d}" if a.due else ""
        owner_txt = a.owner or "unassigned"
        # The link renders as the note title, which is the action's `what`.
        parts.append(f"  - [[{a.note_slug}]] (owner: {owner_txt}{due_txt})")
    parts.append("")

    # Judge failures show on the report, or an outage reads like a quiet
    # week; a clean run prints nothing.
    if themes.judge_failures or themes.judge_cap_hit:
        parts.append("### Run health")
        if themes.judge_failures:
            # Every call failing is flagged as an outage.
            outage_note = (
                " -- OUTAGE: every call failed"
                if themes.judge_failures == themes.judge_calls
                else ""
            )
            parts.append(
                f"- Resolution judge: {themes.judge_failures} of {themes.judge_calls} "
                f"call(s) failed this run{outage_note} "
                "(see changelog kind=theme_judge_error)"
            )
        if themes.judge_cap_hit:
            parts.append(
                f"- Resolution judge: hit its per-run cap ({themes.judge_calls} calls) "
                "before every living theme was checked (see changelog "
                "kind=theme_judge_cap_reached)"
            )
        parts.append("")

    parts.append("### Coverage")
    parts.append(
        f"- Seen {coverage.seen}, extracted {coverage.extracted}, partial {coverage.partial}, "
        f"failed {coverage.failed}, sampled {coverage.sampled}"
    )
    if coverage.filtered_by_reason:
        reasons = ", ".join(f"{k}: {v}" for k, v in sorted(coverage.filtered_by_reason.items()))
        parts.append(f"- Filtered by reason: {reasons}")
    if coverage.sample_miss_count is not None:
        parts.append(f"- Sample audit misses: {coverage.sample_miss_count}")
    parts.append("")

    parts.append("### Quiet accounts")
    qa = diff.quiet_accounts
    parts.append("- 30d: " + (", ".join(f"[[{a}]]" for a in qa.quiet_30) or "none"))
    parts.append("- 60d: " + (", ".join(f"[[{a}]]" for a in qa.quiet_60) or "none"))
    parts.append("- 90d: " + (", ".join(f"[[{a}]]" for a in qa.quiet_90) or "none"))
    parts.append("")

    parts.append("### Contact changes")
    if contact_changes:
        for c in contact_changes:
            acct = f"[[{c.account_slug}]]" if c.account_slug else "(no account)"
            parts.append(f"- {acct}: {c.claim_text} ^[{c.source_slug} @ {c.locator}]")
    else:
        parts.append("- none this window")

    if footnotes:
        parts.append("")
        parts.append("### Notes")
        parts.extend(f"- {note}" for note in footnotes)

    return "\n".join(parts).rstrip() + "\n"


def build_weekly(
    conn,
    llm: LLMClient,
    embedder: Embedder,
    *,
    week: str | None = None,
    as_of: date,
    connector=None,
    arr_by_note_slug: dict[str, float] | None = None,
    region_by_note_slug: dict[str, str] | None = None,
    clock: str = "record",
    prose_model: str | None = None,
    compare_model: str | None = None,
    research: bool = False,
    researcher: Researcher | None = None,
    judge: SupportJudgeFn | None = None,
    propose_done: bool = False,
    done_judge: SupportJudgeFn | None = None,
    public_signals: bool | None = None,
) -> ReportResult:
    """Build, write and return one week's report (module docstring).

    `public_signals`: render "Public signals this week"
    (`portolan.report.public_signals`). `None` follows the
    `PORTOLAN_EXTERNAL_SIGNALS` setting, except that a window which already
    holds public signals always shows them.

    The window is `week`'s Monday to Sunday when `week` is given (then
    `as_of` is replaced), else the seven days ending `as_of`. `clock`
    ("record" or "world") reaches the diff, themes, standing questions and
    the actions feed.

    `research=True` finds candidates and runs one researcher per
    candidate before the prose calls, queueing every card that passes the
    grounding gate; `researcher`/`judge` are injectable and otherwise
    built from settings only when needed. Queued cards for the week are
    rendered on every build, research or not. `propose_done=True` queues
    a "Proposed done" review row for each open action a later vendor-side
    claim answers (`portolan.report.action_done`, judged by `done_judge`).

    `prose_model` goes to every prose call (theme naming, account
    synthesis, standing questions, headline, executive summary); `None`
    keeps `llm`'s default. `compare_model` goes to the theme resolution
    judge (`refresh_themes`, which falls back to its own setting)."""
    if week is not None:
        since, as_of = week_bounds(week)
    else:
        since = as_of - timedelta(days=6)
        # Label by the middle day: a seven-day window spans at most two ISO
        # weeks, and its 4th day is in the one holding the majority.
        week = iso_week_str(since + timedelta(days=3))

    if arr_by_note_slug is None and connector is not None:
        from portolan.report.accounts import account_arr_by_note_slug

        arr_by_crm_id = load_account_arr_via_connector(connector, conn)
        arr_by_note_slug = account_arr_by_note_slug(conn, arr_by_crm_id)
    arr_by_note_slug = arr_by_note_slug or {}

    # Region, like ARR, comes from the CRM export.
    if region_by_note_slug is None and connector is not None:
        region_by_crm_id = load_account_regions_via_connector(connector)
        region_by_note_slug = account_region_by_note_slug(conn, region_by_crm_id)
    region_by_note_slug = region_by_note_slug or {}

    diff = diff_feed.build_diff(conn, since=since, as_of=as_of, clock=clock)
    themes = themes_feed.refresh_themes(
        conn,
        llm,
        embedder,
        since=since,
        as_of=as_of,
        arr_by_note_slug=arr_by_note_slug,
        clock_mode=clock,
        prose_model=prose_model,
        compare_model=compare_model,
    )
    questions = questions_feed.answer_standing_questions(
        conn,
        llm,
        embedder,
        since=since,
        as_of=as_of,
        clock=clock,
        prose_model=prose_model,
        week=week,
    )
    outliers = outliers_feed.list_outliers(conn, since=since, as_of=as_of)
    # No `clock`: a wave is about when things happened in the world.
    waves = waves_feed.compute_waves(
        conn, since=since, as_of=as_of, arr_by_note_slug=arr_by_note_slug
    )
    synthesis = synthesis_feed.refresh_account_synthesis(
        conn,
        llm,
        since=since,
        as_of=as_of,
        arr_by_note_slug=arr_by_note_slug,
        embedder=embedder,
        prose_model=prose_model,
    )
    actions = _actions_feed(conn, since=since, as_of=as_of, clock=clock)
    coverage = _coverage_summary(conn, since=since, as_of=as_of)
    contact_changes = contact_changes_feed(conn, since=since, as_of=as_of)
    public_signal_rows = public_signals_feed.public_signals_feed(
        conn, since=since, as_of=as_of, clock=clock
    )
    if public_signals is None:
        from portolan.config import get_settings

        public_signals = get_settings().external_signals_enabled or bool(public_signal_rows)

    # Ranked themes, for both the executive summary and "What needs attention".
    priorities: list[ThemePriority] = compute_theme_priorities(
        conn,
        themes=themes.themes,
        since=since,
        as_of=as_of,
        arr_by_note_slug=arr_by_note_slug,
        region_by_note_slug=region_by_note_slug,
    )

    # After the feeds candidates read, before the two prose calls (spec:
    # "runs candidates then one researcher per candidate before prose").
    if research:
        from portolan.report.candidates import find_candidates
        from portolan.report.researcher import run_research_candidates

        researcher_instance, judge_instance = _default_researcher_and_judge(
            researcher, judge, embedder=embedder, connector=connector, as_of=as_of
        )
        candidates = find_candidates(
            conn,
            since=since,
            as_of=as_of,
            diff=diff,
            priorities=priorities,
            questions=questions,
            actions=actions,
            arr_by_note_slug=arr_by_note_slug,
            connector=connector,
        )
        research_outcomes = run_research_candidates(
            conn,
            candidates,
            researcher=researcher_instance,
            judge=judge_instance,
            as_of=as_of,
            week=week,
        )
        # The final gate drops a card on a judge error; say so, or a judge
        # outage reads like a quiet week.
        judge_error_slugs = [
            outcome.candidate.slug
            for outcome in research_outcomes
            if outcome.gate is not None
            and not outcome.gate.passed
            and outcome.gate.reason
            and "judge_error" in outcome.gate.reason
        ]
        if judge_error_slugs:
            print(
                f"research: {len(judge_error_slugs)} card(s) dropped for judge_error "
                f"this run ({', '.join(judge_error_slugs)}) -- see changelog "
                "kind=action_card_dropped"
            )

    if propose_done:
        from portolan.report.action_done import outcomes_summary, propose_done_actions

        done_outcomes = propose_done_actions(
            conn,
            judge=done_judge if done_judge is not None else _default_done_judge(),
            embedder=embedder,
            as_of=as_of,
            week=week,
        )
        print(outcomes_summary(done_outcomes))

    # Every build, research or not, so a regen keeps the cards a prior
    # `--research` run queued.
    from portolan.report.researcher import action_card_report_lines

    action_card_lines = action_card_report_lines(conn, week=week)

    # Fills `ReportResult.headline`; `_render_body` does not render it.
    headline_failure_reasons: list[str] = []
    headline = _headline(
        conn,
        llm,
        diff,
        themes,
        questions,
        outliers,
        prose_model=prose_model,
        failure_reasons=headline_failure_reasons,
    )
    headline_reason = headline_failure_reasons[0] if headline_failure_reasons else None

    # Last, since it draws on every feed; a failure leaves the rest of the
    # report intact and its reason in the notes.
    executive_failure_reasons: list[str] = []
    executive_summary = executive_feed.build_executive_summary(
        conn,
        llm,
        since=since,
        as_of=as_of,
        diff=diff,
        themes=themes,
        waves=waves,
        questions=questions,
        outliers=outliers,
        contact_changes=contact_changes,
        actions=actions,
        coverage=coverage,
        arr_by_note_slug=arr_by_note_slug,
        priorities=priorities,
        prose_model=prose_model,
        failure_reasons=executive_failure_reasons,
    )

    body = _render_body(
        week,
        headline,
        diff,
        themes,
        waves,
        questions,
        outliers,
        actions,
        coverage,
        contact_changes,
        clock=clock,
        headline_reason=headline_reason,
        executive_summary=executive_summary,
        priorities=priorities,
        action_card_lines=action_card_lines,
        public_signals=public_signal_rows if public_signals else None,
    )

    slug = report_note_slug(week)
    related = sorted(
        {
            *(t.slug for t in themes.themes),
            *(a for w in waves.waves for a in w.account_slugs),
            *(nc.slug for nc in diff.notes_created),
            *(nc.slug for nc in diff.notes_updated),
            *(o.account_slug for o in outliers if o.account_slug),
            *(o.account_slug for o in contact_changes if o.account_slug),
            *(p.account_slug for p in public_signal_rows if public_signals and p.account_slug),
            *(a.note_slug for a in actions.open),
            *diff.quiet_accounts.quiet_30,
            *(p.account_slug for p in executive_summary.by_account),
        }
    )

    now = clock_module.now()
    note = NoteRow(
        slug=slug,
        type=NoteType.REPORT,
        title=f"Weekly report — {week}",
        description=f"Weekly Portolan report for {week}.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis=body,
        related=related,
    )
    # No reverse `## Related` backlinks: touching every cited note would
    # make next week's diff count them all as updated.
    write_note(conn, note, [], changed_by="report", reason="weekly report", embedder=embedder)

    # Judge health, queryable across weeks. An outage is every call
    # failing; a run with no calls is not one.
    judge_outage = themes.judge_failures == themes.judge_calls and themes.judge_calls > 0
    judge_health = {
        "judge_calls": themes.judge_calls,
        "judge_failures": themes.judge_failures,
        "judge_cap_hit": themes.judge_cap_hit,
        "outage": judge_outage,
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO reports (week, note_slug, judge_health) VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (week) DO UPDATE
            SET note_slug = EXCLUDED.note_slug, judge_health = EXCLUDED.judge_health
            """,
            (week, slug, json.dumps(judge_health)),
        )
    conn.commit()

    return ReportResult(
        week=week,
        since=since,
        as_of=as_of,
        note_slug=slug,
        headline=headline,
        executive_summary=executive_summary,
        diff=diff,
        themes=themes,
        waves=waves,
        priorities=priorities,
        standing_questions=questions,
        outliers=outliers,
        actions=actions,
        coverage=coverage,
        contact_changes=contact_changes,
        public_signals=public_signal_rows if public_signals else [],
        synthesis=synthesis,
        body_md=body,
    )


def render_email(report: ReportResult) -> str:
    """Markdown rendering of an already-built `ReportResult`
    (surfacing-and-weekly-report.md: "Delivery is a rendering of that
    note: email, Slack, PDF. The note is the truth; the email is a
    copy.")."""
    header = (
        f"# Weekly report — {report.week}\n\n"
        f"_{report.since:%Y-%m-%d} to {report.as_of:%Y-%m-%d}_\n\n"
    )
    return header + report.body_md


# --------------------------------------------------------------------------
# The research harness: only the feeds `find_candidates` reads, then the
# researcher loop, with no report written, so one stage can be tested cheaply.
# --------------------------------------------------------------------------


def _default_researcher_and_judge(
    researcher: Researcher | None,
    judge: SupportJudgeFn | None,
    *,
    embedder: Embedder,
    connector: Any,
    as_of: date,
) -> tuple[Researcher, SupportJudgeFn]:
    """Fills in whichever of `researcher`/`judge` was not given, for both
    `build_weekly` and `run_research_harness`. The judge is built first
    and handed to a default `AgentSDKResearcher`, so `submit_card` and the
    final gate use the same judge."""
    if researcher is not None and judge is not None:
        return researcher, judge

    import logging

    from portolan.config import get_settings
    from portolan.llm.factory import get_llm
    from portolan.report.researcher import AgentSDKResearcher
    from portolan.report.support_judge import make_llm_support_judge
    from portolan.store import LocalStore

    settings = get_settings()
    if judge is None:
        judge = make_llm_support_judge(get_llm(settings, provider="agent_sdk"))
    if researcher is None:
        # `fallback_model` must be a model the local Claude Code CLI can run;
        # any other `COMPARE_MODEL` id is skipped with a warning.
        compare_model = settings.compare_model
        fallback_model = None
        if compare_model and compare_model.lower().startswith("claude-"):
            fallback_model = compare_model
        elif compare_model:
            logging.getLogger(__name__).warning(
                "COMPARE_MODEL=%r is not a Claude model id; the action researcher "
                "will run with no fallback_model (Sonnet rate-limited or "
                "unavailable will simply fail that candidate rather than degrade "
                "to a broken fallback)",
                compare_model,
            )
        researcher = AgentSDKResearcher(
            database_url=settings.database_url.get_secret_value(),
            embedder=embedder,
            store=LocalStore(settings.store_root),
            tenant=settings.tenant,
            fallback_model=fallback_model,
            connector=connector,
            as_of=as_of,
            judge=judge,
        )
    return researcher, judge


def _default_done_judge() -> SupportJudgeFn:
    """The "does this later claim answer the action" judge, on the same
    agent-SDK path the researcher's support judge uses."""
    from portolan.config import get_settings
    from portolan.llm.factory import get_llm
    from portolan.report.action_done import make_llm_done_judge

    return make_llm_done_judge(get_llm(get_settings(), provider="agent_sdk"))


class UnknownCandidateSlugError(ValueError):
    """Raised by `run_research_harness` when a requested candidate slug is
    not among the week's candidates: `unknown` names the misses, `valid`
    lists every real candidate slug, so the CLI can print both and exit
    non-zero rather than look like an empty week."""

    def __init__(self, *, unknown: list[str], valid: list[str]) -> None:
        self.unknown = unknown
        self.valid = valid
        super().__init__(
            f"unknown candidate slug(s): {', '.join(unknown)}; "
            f"valid candidate slugs: {', '.join(valid) if valid else '(none this week)'}"
        )


def run_research_harness(
    conn,
    llm: LLMClient,
    embedder: Embedder,
    *,
    week: str,
    clock: str = "record",
    connector: Any = None,
    candidate_slugs: list[str] | None = None,
    researcher: Researcher | None = None,
    judge: SupportJudgeFn | None = None,
    prose_model: str | None = None,
    compare_model: str | None = None,
) -> tuple[list[CandidateResearchOutcome], ThemeReport]:
    """The `research` CLI's pipeline: the feeds `find_candidates` reads
    (the same calls `build_weekly` makes), `find_candidates`, a filter to
    `candidate_slugs` (`None` means all; an unknown slug raises
    `UnknownCandidateSlugError`), then `run_research_candidates`. No
    report is written. `researcher`/`judge` are injectable; `None` builds
    the defaults. Returns the outcomes and the `ThemeReport`, whose judge
    counts the CLI prints."""
    from portolan.report.candidates import find_candidates
    from portolan.report.researcher import run_research_candidates

    since, as_of = week_bounds(week)

    arr_by_crm_id = load_account_arr_via_connector(connector, conn)
    arr_by_note_slug = account_arr_by_note_slug(conn, arr_by_crm_id)
    region_by_crm_id = load_account_regions_via_connector(connector)
    region_by_note_slug = account_region_by_note_slug(conn, region_by_crm_id)

    diff = diff_feed.build_diff(conn, since=since, as_of=as_of, clock=clock)
    themes = themes_feed.refresh_themes(
        conn,
        llm,
        embedder,
        since=since,
        as_of=as_of,
        arr_by_note_slug=arr_by_note_slug,
        clock_mode=clock,
        prose_model=prose_model,
        compare_model=compare_model,
    )
    questions = questions_feed.answer_standing_questions(
        conn,
        llm,
        embedder,
        since=since,
        as_of=as_of,
        clock=clock,
        prose_model=prose_model,
        week=week,
    )
    actions = _actions_feed(conn, since=since, as_of=as_of, clock=clock)
    priorities = compute_theme_priorities(
        conn,
        themes=themes.themes,
        since=since,
        as_of=as_of,
        arr_by_note_slug=arr_by_note_slug,
        region_by_note_slug=region_by_note_slug,
    )

    candidates = find_candidates(
        conn,
        since=since,
        as_of=as_of,
        diff=diff,
        priorities=priorities,
        questions=questions,
        actions=actions,
        arr_by_note_slug=arr_by_note_slug,
        connector=connector,
    )
    if candidate_slugs is not None:
        # Checked before any researcher session or judge call is spent.
        valid_slugs = [c.slug for c in candidates]
        valid_slug_set = set(valid_slugs)
        unknown = [slug for slug in candidate_slugs if slug not in valid_slug_set]
        if unknown:
            raise UnknownCandidateSlugError(unknown=unknown, valid=valid_slugs)
        wanted = set(candidate_slugs)
        candidates = [c for c in candidates if c.slug in wanted]

    researcher_instance, judge_instance = _default_researcher_and_judge(
        researcher, judge, embedder=embedder, connector=connector, as_of=as_of
    )

    outcomes = run_research_candidates(
        conn,
        candidates,
        researcher=researcher_instance,
        judge=judge_instance,
        as_of=as_of,
        week=week,
    )
    return outcomes, themes


def _research_result_line(outcome: CandidateResearchOutcome) -> str:
    """One summary line for one candidate: slug, rule, verdict, evidence
    and findings counts, `submit_card` calls, tool calls, gate outcome."""
    card = outcome.card
    diag = outcome.diagnostics
    if card is not None:
        verdict = card.verdict
        n_evidence = len(card.evidence)
        n_findings = len(card.findings)
    else:
        verdict = f"no_card:{diag.outcome}" if diag is not None else "no_card:unknown"
        n_evidence = 0
        n_findings = 0

    trail = diag.trail if diag is not None else None
    submit_card_calls = sum(1 for t in trail.tool_calls if t.tool == "submit_card") if trail else 0
    tool_calls = diag.tool_calls if diag is not None else 0

    if outcome.gate is None:
        detail = f": {diag.detail}" if diag is not None and diag.detail else ""
        gate_outcome = f"dropped:no_card{detail}"
    elif outcome.gate.passed:
        gate_outcome = "queued" if outcome.queued else "already_queued"
    else:
        gate_outcome = f"dropped:{outcome.gate.reason}"

    return (
        f"{outcome.candidate.slug}  rule={outcome.candidate.rule}  verdict={verdict}  "
        f"evidence={n_evidence}  findings={n_findings}  submit_card_calls={submit_card_calls}  "
        f"tool_calls={tool_calls}  gate={gate_outcome}"
    )


def _print_research_results(outcomes: list[CandidateResearchOutcome]) -> None:
    if not outcomes:
        print("no candidates matched")
        return
    for outcome in outcomes:
        print(_research_result_line(outcome))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    from typing import get_args

    from portolan.config import Settings

    parser = argparse.ArgumentParser(prog="python -m portolan.report")
    sub = parser.add_subparsers(dest="command", required=True)
    weekly_parser = sub.add_parser("weekly")
    # Exactly one: `build_weekly` ignores `as_of` whenever `week` is given.
    as_of_or_week = weekly_parser.add_mutually_exclusive_group(required=True)
    as_of_or_week.add_argument("--as-of", default=None)
    as_of_or_week.add_argument("--week", default=None)
    weekly_parser.add_argument(
        "--llm",
        # Read off the `Settings.llm_provider` Literal so the lists cannot drift.
        choices=list(get_args(Settings.model_fields["llm_provider"].annotation)),
        default=None,
        help=(
            "LLM provider for this report's model calls (theme naming, standing "
            "questions, the headline, the executive summary, account synthesis). "
            "Omit to use this deployment's own LLM_PROVIDER (.env) via "
            "portolan.llm.get_llm."
        ),
    )
    weekly_parser.add_argument(
        "--prose-model",
        default=None,
        help=(
            "Override the model id used for every prose call this report makes "
            "(the headline, the executive summary, theme naming, account "
            "synthesis, standing questions) -- e.g. --llm agent_sdk --prose-model "
            "claude-sonnet-5 to run every prose call through the Claude Agent SDK "
            "with no OpenRouter call at all. Omit to use STRONG_MODEL (falling "
            "back to EXTRACTION_MODEL when unset), this CLI's existing default."
        ),
    )
    weekly_parser.add_argument("--embed", choices=["fake", "fastembed"], default="fake")
    weekly_parser.add_argument("--corpus", type=Path, default=Path("corpus/out"))
    weekly_parser.add_argument("--out", type=Path, default=Path("var/reports"))
    weekly_parser.add_argument(
        "--clock",
        choices=["record", "world"],
        default="record",
        help=(
            "'record' (default) keys the diff's created/updated split on "
            "notes.created/updated (wall-clock write time). 'world' keys it on "
            "claims.recorded (the world date from the source) instead, so a "
            "backfilled corpus can still produce the correct report for a past "
            "--as-of/--week."
        ),
    )
    weekly_parser.add_argument(
        "--research",
        action="store_true",
        help=(
            "Run the action researcher (01-spec/actions-and-owners-2026-09-19.md "
            "Phase 2) before this run's prose: one Claude agent-SDK session per "
            "candidate, gated through the grounding gate, queued as review_queue "
            "action_card rows. Off by default -- without it no researcher runs, "
            "though cards an earlier --research run queued still appear in the report."
        ),
    )
    weekly_parser.add_argument(
        "--propose-done",
        action="store_true",
        help=(
            "Before prose, queue a 'Proposed done' review row for each open action a "
            "later vendor-side claim answers (checklist item 24). One judge call per "
            "candidate claim, capped per action. Off by default."
        ),
    )

    from portolan.report import monday

    monday.add_parser(sub)

    # `propose-done`: the same pass for one week, with no report written.
    done_parser = sub.add_parser("propose-done")
    done_parser.add_argument(
        "--week",
        required=True,
        help="ISO week, e.g. 2026-W36; claims after its Sunday are ignored.",
    )
    done_parser.add_argument("--embed", choices=["fake", "fastembed"], default="fake")
    done_parser.add_argument(
        "--action",
        action="append",
        type=int,
        default=None,
        dest="action_ids",
        metavar="ID",
        help="Check only this action id (repeatable). Default: every open action.",
    )
    done_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the claims that would be judged for each action; no judge call, no writes.",
    )

    # `research`: the researcher alone, no report (`run_research_harness`).
    research_parser = sub.add_parser("research")
    research_parser.add_argument(
        "--week", required=True, help="ISO week, e.g. 2026-W36 (same shape as `weekly --week`)."
    )
    research_parser.add_argument(
        "--llm",
        choices=list(get_args(Settings.model_fields["llm_provider"].annotation)),
        default=None,
        help=(
            "LLM provider for the feeds find_candidates itself reads (theme naming, "
            "standing questions) -- the researcher session always runs through the "
            "Claude agent SDK regardless of this flag. Omit to use this deployment's "
            "own LLM_PROVIDER (.env) via portolan.llm.get_llm."
        ),
    )
    research_parser.add_argument("--embed", choices=["fake", "fastembed"], default="fake")
    research_parser.add_argument("--corpus", type=Path, default=Path("corpus/out"))
    research_parser.add_argument(
        "--clock",
        choices=["record", "world"],
        default="record",
        help="Same meaning as `weekly --clock` -- see that flag's own help.",
    )
    research_parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        dest="candidate_slugs",
        metavar="SLUG",
        help=(
            "Research only this candidate slug (repeatable). Exactly one of "
            "--candidate/--all is required."
        ),
    )
    research_parser.add_argument(
        "--all",
        action="store_true",
        help="Research every candidate find_candidates finds for this week.",
    )
    return parser


def _build_llm(kind: str | None, corpus_path: Path, settings):
    """The report's `LLMClient`, through `portolan.llm.factory.get_llm`.

    `kind=None` uses the deployment's `LLM_PROVIDER`. An explicit `kind`
    overrides only the provider: `Settings(llm_provider=kind)` still reads
    tenant, API key and model ids from the environment/`.env`.

    `fake` never goes through `get_llm`: it is a report-CLI-only replay of
    the corpus's planted facts (`FakePlacements(corpus_path)`).
    """
    if kind == "fake":
        from portolan.fake_extractor import FakePlacements, build_fake_llm_client

        return build_fake_llm_client(FakePlacements(corpus_path), tenant=settings.tenant)

    from portolan.config import Settings
    from portolan.llm.factory import get_llm

    effective_settings = settings if kind is None else Settings(llm_provider=kind)
    return get_llm(effective_settings)


def _build_embedder(kind: str) -> Embedder:
    if kind == "fake":
        from portolan.search.embed import FakeEmbedder

        return FakeEmbedder()
    from portolan.search.embed import FastEmbedEmbedder

    return FastEmbedEmbedder()


def _print_usage_summary(
    conn,
    *,
    tenant: str,
    since_ts,
    extraction_model: str,
    strong_model: str,
    themes: ThemeReport,
) -> None:
    """Print the two model ids the caller names (whether or not either
    was called), the resolution judge's call/failure/cap counts (always,
    so zero failures is visible too), and a per-model breakdown of the
    `usage_events` rows for `surface='report'`, this tenant, since
    `since_ts`."""
    print(f"Models: extraction={extraction_model} strong={strong_model}")
    cap_note = ", cap hit" if themes.judge_cap_hit else ""
    outage_note = (
        ", OUTAGE (every call failed)"
        if themes.judge_failures and themes.judge_failures == themes.judge_calls
        else ""
    )
    print(
        f"Resolution judge: {themes.judge_calls} call(s), {themes.judge_failures} "
        f"failed{cap_note}{outage_note}"
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_key, COUNT(*), COALESCE(SUM(input_tokens), 0),
                   COALESCE(SUM(output_tokens), 0), COALESCE(SUM(cost_estimate), 0)
            FROM usage_events
            WHERE tenant = %s AND surface = 'report' AND created >= %s
            GROUP BY model_key
            ORDER BY model_key
            """,
            (tenant, since_ts),
        )
        rows = cur.fetchall()
    if not rows:
        print("Usage (surface=report): no model calls recorded for this run")
        return
    print("Usage by model (surface=report):")
    for model_key, calls, input_tokens, output_tokens, cost in rows:
        print(
            f"  {model_key}: {calls} calls, {input_tokens} input tokens, "
            f"{output_tokens} output tokens, ${float(cost):.6f}"
        )


def _run_research_cli(args: argparse.Namespace) -> int:
    from portolan import db
    from portolan.config import get_settings
    from portolan.connectors.file_connector import FileConnector

    if bool(args.all) == bool(args.candidate_slugs):
        print(
            "research: pass exactly one of --candidate (repeatable) or --all",
            file=sys.stderr,
        )
        return 2

    settings = get_settings()
    effective_provider = args.llm or settings.llm_provider
    print(f"Provider: {effective_provider}")
    run_started_at = clock_module.now()

    conn = db.connect(autocommit=True)
    try:
        db.apply_migrations(conn)
        connector = FileConnector(args.corpus)
        llm = _build_llm(args.llm, args.corpus, settings)
        embedder = _build_embedder(args.embed)

        candidate_slugs = None if args.all else list(args.candidate_slugs)
        try:
            outcomes, themes = run_research_harness(
                conn,
                llm,
                embedder,
                week=args.week,
                clock=args.clock,
                connector=connector,
                candidate_slugs=candidate_slugs,
            )
        except UnknownCandidateSlugError as exc:
            print(
                f"research: unknown candidate slug(s): {', '.join(exc.unknown)}",
                file=sys.stderr,
            )
            print(
                "research: valid candidate slugs for "
                f"{args.week}: {', '.join(exc.valid) if exc.valid else '(none)'}",
                file=sys.stderr,
            )
            return 2
        _print_research_results(outcomes)
        _print_usage_summary(
            conn,
            tenant=settings.tenant,
            since_ts=run_started_at,
            extraction_model=settings.extraction_model,
            strong_model=settings.strong_model or settings.extraction_model,
            themes=themes,
        )
    finally:
        conn.close()
    return 0


def _run_propose_done_cli(args: argparse.Namespace, *, judge: SupportJudgeFn | None = None) -> int:
    from portolan import db
    from portolan.report.action_done import outcome_line, outcomes_summary, propose_done_actions

    _since, as_of = week_bounds(args.week)
    conn = db.connect(autocommit=True)
    try:
        db.apply_migrations(conn)
        if not args.dry_run and judge is None:
            judge = _default_done_judge()
        outcomes = propose_done_actions(
            conn,
            judge=None if args.dry_run else judge,
            embedder=_build_embedder(args.embed),
            as_of=as_of,
            week=args.week,
            action_ids=args.action_ids,
            dry_run=args.dry_run,
        )
    finally:
        conn.close()
    for outcome in outcomes:
        print(outcome_line(outcome))
    print(outcomes_summary(outcomes))
    return 0


def main(argv: list[str] | None = None) -> int:
    from portolan import db
    from portolan.config import get_settings
    from portolan.connectors.file_connector import FileConnector

    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_arg_parser().parse_args(argv)

    if args.command == "research":
        return _run_research_cli(args)
    if args.command == "propose-done":
        return _run_propose_done_cli(args)
    if args.command == "monday":
        from portolan.report import monday

        return monday.run_cli(args)

    # Without `--as-of`, `--week` was given; `build_weekly` derives the same `as_of` from it.
    as_of = date.fromisoformat(args.as_of) if args.as_of is not None else week_bounds(args.week)[1]
    settings = get_settings()
    # Prose uses `--prose-model`, else STRONG_MODEL, else EXTRACTION_MODEL.
    # No `compare_model` is passed: the resolution judge keeps its own setting.
    prose_model = args.prose_model or settings.strong_model or settings.extraction_model
    # Printed before any model call, so a failed run still names them.
    effective_provider = args.llm or settings.llm_provider
    print(f"Provider: {effective_provider}  Prose model: {prose_model}")
    run_started_at = clock_module.now()
    # Autocommit keeps the connection idle between statements, so each
    # `with conn.transaction():` is a real BEGIN/COMMIT, not a SAVEPOINT
    # (`test_weekly_main_opens_its_connection_with_autocommit`).
    conn = db.connect(autocommit=True)
    try:
        db.apply_migrations(conn)
        connector = FileConnector(args.corpus)
        llm = _build_llm(args.llm, args.corpus, settings)
        embedder = _build_embedder(args.embed)
        result = build_weekly(
            conn,
            llm,
            embedder,
            week=args.week,
            as_of=as_of,
            connector=connector,
            clock=args.clock,
            prose_model=prose_model,
            research=args.research,
            propose_done=args.propose_done,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        out_path = args.out / f"report-{result.week}.md"
        out_path.write_text(render_email(result))
        print(f"wrote {out_path}")
        _print_usage_summary(
            conn,
            tenant=settings.tenant,
            since_ts=run_started_at,
            extraction_model=settings.extraction_model,
            strong_model=prose_model,
            themes=result.themes,
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
