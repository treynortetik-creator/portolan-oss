"""The weekly report's executive summary (`01-spec/surfacing-and-weekly-report.md`).

`build_executive_summary` is a strong-model pass over the feeds
`build_weekly` already computed (diff, themes, waves, standing questions,
outliers, contact changes, actions, coverage), each active account's
current synthesis, and every claim recorded in the window
(`_window_claims`), since the other feeds carry only aggregates. The
model writes four parts: `this_week`, `attention`, `by_account` and
`suggested_actions` (`EXECUTIVE_SYSTEM` sets their sizes).

Each sentence, bullet or paragraph is gated on its own, so one bad item
never costs the others (`_gate_sentence`): stray `[[slug]]` stripped,
citation markers must be in the payload, numerals must be in the
`NumberFacts` table (spec "Numbers rule"), and the resolved-language
guard applies. A `by_account` paragraph for an account not active this
window is dropped.

`MAX_RENDERED_LINES` (80) is a backstop against a runaway response,
counted over the kept items (`this_week` counts as one) after gating.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date

import psycopg
from pydantic import BaseModel, ConfigDict, Field

from portolan.llm.base import LLMClient
from portolan.notes import read_note
from portolan.notes.render import plain_text
from portolan.report import diff as diff_feed
from portolan.report import themes as themes_feed
from portolan.report.accounts import home_account_slug
from portolan.report.models import (
    AccountParagraph,
    ActionsFeed,
    CoverageSummary,
    DiffFeed,
    ExecutiveSummary,
    OutlierClaim,
    StandingQuestionAnswer,
    ThemePriority,
    ThemeReport,
    WavesFeed,
)
from portolan.report.numbers import NumberFacts, check_numerals
from portolan.report.priorities import add_priority_facts, format_priority_stats
from portolan.report.resolved_language import resolved_language_rejected
from portolan.report.util import citation_markers_in, validate_citations
from portolan.util import clock as clock_module

__all__ = ["EXECUTIVE_SYSTEM", "MAX_RENDERED_LINES", "build_executive_summary"]

MAX_RENDERED_LINES = 80

EXECUTIVE_SYSTEM = (
    "You write the executive summary at the top of a weekly customer-signals "
    "report, for a business reader (a VP of Sales) who has two minutes and no "
    "patience for jargon or repetition. Target the WHOLE summary at roughly "
    "500-650 words total (about a two-minute read) -- when there are many "
    "active accounts, that means each one gets tighter, not that some get "
    "skipped. You are given the WHOLE week's material as compact JSON: every "
    "individual claim recorded this week (fact type, account, sentence, "
    "verbatim quote when there is one, source, date, locator, and its own "
    "citation marker), the mechanically-computed number table ('figures'), "
    "themes by movement, waves, open/overdue/resolved actions, outliers, "
    "standing-question answers, contact changes, and the current per-account "
    "synthesis for every active account. Write four parts, plain business "
    "English, no jargon, and reply as JSON with exactly these keys:\n"
    "- 'this_week': 3 to 5 sentences giving the overall shape of the week -- "
    "the 2 or 3 biggest patterns or swings and why they matter. This is the "
    "headline picture, not a roll call of every account; save individual "
    "account facts for 'attention' and 'by_account' below rather than "
    "repeating them here near-verbatim.\n"
    "- 'attention': 3 to 7 bullet sentences, each one CONCRETE finding that "
    "names the account and the specific fact (never 'several accounts "
    "raised concerns' -- name them), with a citation. When the payload "
    "includes a 'priorities' list, it is already ranked highest-priority "
    "first (most accounts touched, then most ARR touched, then how urgent "
    "the movement is) -- cover the themes it names in that same order "
    "before any theme not on that list, so this section reads in the "
    "priority the numbers already established rather than a different "
    "order of your own choosing.\n"
    "- 'by_account': one entry per ACTIVE account (an account this week's "
    "claims actually touch), most important first (by risk and how much "
    "changed, not simply by ARR -- a small account in crisis outranks a "
    "large quiet one). Each entry is 1 to 3 SHORT sentences and MUST STAND "
    "ALONE -- a reader who skips straight to 'by_account' without reading "
    "'attention' first still has to understand what happened. Open by "
    "naming the account and the concrete fact in plain words (what "
    "happened, briefly stated), THEN add the context and what it means -- "
    "the theme or trend it sits inside, the risk, or the next step. Never "
    "open a sentence with a bare pronoun or a fragment that assumes the "
    "reader already has the context -- 'This flip...', 'This third "
    "repeated ask...', 'The change lands...' are all wrong; say what "
    "'this'/'the change' actually was, every time. You do not need to "
    "repeat the citation marker a second time when 'attention' already "
    "cited the same fact, but the sentence itself must still say the "
    "fact, not just point at it -- a meta-reference phrase like 'see "
    "above'/'as noted'/'as mentioned' with no fact of its own is never "
    "allowed. 'account_slug' must be copied EXACTLY, character for "
    "character, from this account's own 'slug' field in the payload's "
    "'accounts' list -- never invented, never an account not in that "
    "list.\n"
    "- 'suggested_actions': 2 to 6 bullet sentences, each tied to a specific "
    "fact you cite -- a concrete next step, not generic advice. When a suggested "
    "action matches or extends something already in the JSON's own 'open_actions' "
    "list, name that action's 'owner' by name (never 'unassigned', and never a name "
    "not in 'open_actions') so a reader knows who already owns it -- but never invent "
    "an owner for an action that isn't in that list.\n\n"
    "Rules that apply to every sentence in every part: every sentence must "
    "stand on its own -- say what happened in plain words wherever you "
    "mention it, never a bare pronoun or fragment ('this flip', 'the "
    "change', 'that request') that only makes sense next to a different "
    "sentence. The one thing you should NOT repeat every time the same "
    "fact comes up is its citation marker -- cite a given fact in full the "
    "first time it appears, and a plain, still-complete sentence (no "
    "citation needed) is fine anywhere after that. That distinction is the "
    "fix for 'everything repeats': the FACT gets stated in full wherever "
    "it's relevant, only the CITATION doesn't need repeating. Cite claims "
    "with their exact marker where the JSON gives one -- reproduce "
    "it exactly as given, character for character, INCLUDING the leading "
    "'^' -- '^[source @ locator]', never 'source @ locator' or '[source @ "
    "locator]' without the caret. Every number you state must come from "
    "the JSON's own 'figures' table or appear verbatim elsewhere in the "
    "JSON -- never compute, round, sum, or estimate a number yourself. "
    "Write account and theme names as plain words only -- never wrap one "
    "in double square brackets like '[[slug]]'; the report links every "
    "name mechanically, so your sentences need only say the name once, in "
    "prose. Absolute dates only, never 'this week' or 'recently'. Never "
    "say a theme is resolved, or use the word 'resolution', for a theme "
    "whose own 'movement' value is not exactly 'resolved'. Never write the "
    "bare phrase 'N signals' (or 'N accounts said...') without also saying "
    "what they are -- name the account(s) and the fact. Do not simply "
    "restate a theme's own name as if that were the finding; say what is "
    "actually happening."
)


class _AccountParagraphLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_slug: str
    paragraph: str


class _ExecutiveSummaryLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    this_week: list[str] = Field(default_factory=list)
    attention: list[str] = Field(default_factory=list)
    by_account: list[_AccountParagraphLLM] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)


def _executive_summary_json_schema() -> dict:
    return _ExecutiveSummaryLLM.model_json_schema()


def _queue_executive_rejection(
    conn: psycopg.Connection, *, surface: str, detail: str, reason: str = "other"
) -> None:
    payload = {"reason": reason, "surface": surface, "note_slug": None, "detail": detail}
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('other', %s::jsonb)",
            (json.dumps(payload),),
        )
    conn.commit()


def _window_claims(conn: psycopg.Connection, *, since: date, as_of: date) -> list[dict]:
    """Every non-archived claim recorded this window, by world date
    (`claims.recorded`) whatever the report's `--clock`.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, c.fact_type, c.claim_text, c.quote, c.source_slug, c.locator,
                   c.note_slug, s.source_type, s.source_date
            FROM claims c
            JOIN sources s ON s.slug = c.source_slug
            WHERE c.archived = false
              AND {clock_module.world_date_sql("c.recorded")} BETWEEN %s AND %s
            ORDER BY c.recorded, c.id
            """,
            (since, as_of),
        )
        rows = cur.fetchall()

    out: list[dict] = []
    for row in rows:
        (
            cid,
            fact_type,
            claim_text,
            quote,
            source_slug,
            locator,
            note_slug,
            source_type,
            src_date,
        ) = row
        account = home_account_slug(
            conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
        )
        out.append(
            {
                "claim_id": cid,
                "fact_type": fact_type,
                "account": account,
                "sentence": plain_text(claim_text),
                "quote": quote,
                "source_type": source_type,
                "source_date": src_date.isoformat() if src_date else None,
                "locator": locator,
                "citation": f"^[{source_slug} @ {locator}]",
            }
        )
    return out


def _account_context(
    conn: psycopg.Connection, account_slugs: list[str], arr_by_note_slug: dict[str, float]
) -> list[dict]:
    """Name, ARR (when known) and current synthesis text for each active
    account whose note exists, for the model's "By account" paragraphs."""
    out: list[dict] = []
    for slug in account_slugs:
        try:
            note, _claims = read_note(conn, slug)
        except ValueError:
            continue
        out.append(
            {
                "slug": slug,
                "name": note.title,
                "arr": arr_by_note_slug.get(slug),
                "current_synthesis": note.synthesis or "",
            }
        )
    return out


def _gate_sentence(
    conn: psycopg.Connection,
    raw_sentence: str,
    *,
    allowed_markers: set[tuple[str, str]],
    facts: NumberFacts,
    themes: ThemeReport,
    surface: str,
    grounded_texts: list[str] | None = None,
) -> str | None:
    """One sentence/bullet/paragraph through every mechanical gate: stray
    `[[slug]]` stripped, citation validation (including a caret-less
    marker), the numeral rule, and the resolved-language guard. Returns
    the cleaned text, or `None` when rejected (logged to `review_queue`
    under `surface`) or empty.
    """
    # Imported here: `weekly` imports this module at load time.
    from portolan.report.weekly import _MALFORMED_CITATION_RE, _strip_stray_wikilinks

    sentence = _strip_stray_wikilinks(raw_sentence)
    if not sentence:
        return None
    if validate_citations(sentence, allowed_markers) is not None:
        _queue_executive_rejection(
            conn, surface=surface, detail=f"unsupported citation marker: {sentence!r}"
        )
        return None
    if _MALFORMED_CITATION_RE.search(sentence):
        _queue_executive_rejection(
            conn, surface=surface, detail=f"malformed citation marker: {sentence!r}"
        )
        return None
    numeral_violations = check_numerals(sentence, facts)
    if numeral_violations:
        _queue_executive_rejection(
            conn,
            surface=surface,
            detail=f"numeral(s) not in the supplied figures: {', '.join(numeral_violations)}",
        )
        return None
    if resolved_language_rejected(sentence, themes.themes, grounded_texts=grounded_texts or ()):
        _queue_executive_rejection(
            conn,
            surface=surface,
            detail=sentence,
            reason="resolved_language_without_a_resolved_theme",
        )
        return None
    return sentence


def build_executive_summary(
    conn: psycopg.Connection,
    llm: LLMClient,
    *,
    since: date,
    as_of: date,
    diff: DiffFeed,
    themes: ThemeReport,
    waves: WavesFeed,
    questions: list[StandingQuestionAnswer],
    outliers: list[OutlierClaim],
    contact_changes: list[OutlierClaim],
    actions: ActionsFeed,
    coverage: CoverageSummary,
    arr_by_note_slug: dict[str, float] | None = None,
    priorities: list[ThemePriority] | None = None,
    prose_model: str | None = None,
    failure_reasons: list[str] | None = None,
) -> ExecutiveSummary:
    """Build the report's executive summary (module docstring). A failed
    model call, an all-rejected reply or an over-cap summary returns an
    empty `ExecutiveSummary`; when `failure_reasons` is a list, a
    plain-language reason is appended and copied to `failure_reason`.

    `priorities` (`compute_theme_priorities`, already ranked) is handed to
    the model in order, so its `attention` bullets follow the ranking, and
    registered with `add_priority_facts` so a ranked theme's account count
    or ARR passes the numeral gate. `None`/empty adds neither.
    """
    arr_by_note_slug = arr_by_note_slug or {}
    claims = _window_claims(conn, since=since, as_of=as_of)

    account_slugs = sorted({c["account"] for c in claims if c["account"]})
    accounts = _account_context(conn, account_slugs, arr_by_note_slug)

    facts = NumberFacts()
    facts.add("notes_created", len(diff.notes_created))
    facts.add("notes_updated", len(diff.notes_updated))
    facts.add("quiet_30_count", len(diff.quiet_accounts.quiet_30))
    facts.add("quiet_60_count", len(diff.quiet_accounts.quiet_60))
    facts.add("quiet_90_count", len(diff.quiet_accounts.quiet_90))
    # Thresholds and the window length are facts too, so "quiet for 30
    # days" passes the numeral gate.
    for threshold_days in diff_feed.QUIET_ACCOUNT_THRESHOLDS:
        facts.add(f"quiet_threshold_{threshold_days}_days", threshold_days)
    facts.add("window_days", (as_of - since).days + 1)
    facts.add("theme_resolved_after_weeks", themes_feed.THEME_RESOLVED_AFTER_WEEKS)
    facts.add("outliers_count", len(outliers))
    facts.add("contact_changes_count", len(contact_changes))
    facts.add("actions_open_count", len(actions.open))
    facts.add("actions_overdue_count", len(actions.overdue))
    facts.add("actions_resolved_count", len(actions.resolved_this_week))
    # Per-owner open/overdue counts, keyed by crm_id (display names can
    # collide), so a sentence can cite "N open actions for <owner>".
    open_owner_ids = (a.owner_crm_id for a in actions.open if a.owner_crm_id)
    for owner_crm_id, count in Counter(open_owner_ids).items():
        facts.add(f"owner-{owner_crm_id}:open_count", count)
    for owner_crm_id, count in Counter(
        a.owner_crm_id for a in actions.overdue if a.owner_crm_id
    ).items():
        facts.add(f"owner-{owner_crm_id}:overdue_count", count)
    facts.add("coverage_seen", coverage.seen)
    facts.add("coverage_extracted", coverage.extracted)
    for t in themes.themes:
        facts.add(f"{t.slug}:signal_count", t.signal_count)
        facts.add(f"{t.slug}:linked_arr", t.linked_arr, f"{t.linked_arr:,.0f}")
    for w in waves.waves:
        key = f"wave-{w.week_of:%Y-%m-%d}-{w.fact_type}-{w.product_family}"
        facts.add(f"{key}:account_count", len(w.account_slugs))
        facts.add(f"{key}:linked_arr", w.linked_arr, f"{w.linked_arr:,.0f}")
    for acct in accounts:
        if acct["arr"] is not None:
            facts.add(f"{acct['slug']}:arr", acct["arr"], f"{acct['arr']:,.0f}")
    if priorities:
        add_priority_facts(facts, priorities)

    # A claim's numbers are licensed only in a sentence that cites that
    # claim, never attached to a different claim's citation.
    for c in claims:
        text = c["sentence"] + (f" {c['quote']}" if c["quote"] else "")
        facts.harvest(text, marker=(_source_slug_of(c), c["locator"]))
    # Standing-question answers and syntheses have no single claim to
    # bind to, so their numbers are licensed anywhere.
    facts.harvest(json.dumps([q.answer_md for q in questions]))
    facts.harvest(json.dumps([a["current_synthesis"] for a in accounts]))

    payload = {
        "window": {"since": since.isoformat(), "as_of": as_of.isoformat()},
        "claims": [{k: v for k, v in c.items() if k != "claim_id"} for c in claims],
        "accounts": accounts,
        "themes": [
            {
                "name": t.name,
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
        "waves": [
            {
                "week_of": w.week_of.isoformat(),
                "fact_type": w.fact_type,
                "product_family": w.product_family,
                "accounts": w.account_slugs,
                "linked_arr": w.linked_arr,
            }
            for w in waves.waves
        ],
        "standing_questions": [{"question": q.question, "answer": q.answer_md} for q in questions],
        "outliers": [
            {
                "account": o.account_slug,
                "sentence": o.claim_text,
                "citation": f"^[{o.source_slug} @ {o.locator}]",
            }
            for o in outliers
        ],
        "contact_changes": [
            {
                "account": c.account_slug,
                "sentence": c.claim_text,
                "citation": f"^[{c.source_slug} @ {c.locator}]",
            }
            for c in contact_changes
        ],
        "actions_open": len(actions.open),
        "actions_overdue": len(actions.overdue),
        "actions_resolved_this_week": len(actions.resolved_this_week),
        # Lets a suggested action name the action's real owner.
        "open_actions": [
            {
                "what": a.what,
                "owner": a.owner or "unassigned",
                "due": a.due.isoformat() if a.due else None,
            }
            for a in actions.open
        ],
        "coverage": {"seen": coverage.seen, "extracted": coverage.extracted},
    }
    if priorities:
        # List order is the ranking; the prompt asks for that order.
        payload["priorities"] = [
            {
                "theme": p.theme_name,
                "movement": p.movement,
                "account_count": p.account_count,
                "arr": p.arr,
                "regions": p.regions,
                "stats": format_priority_stats(p),
            }
            for p in priorities
        ]
    payload["figures"] = {name: rendering for name, (_v, rendering) in facts.items()}
    payload_text = json.dumps(payload, default=str)
    allowed_markers = citation_markers_in(payload_text)
    grounded_texts = [t for c in claims for t in (c["sentence"], c["quote"]) if t]

    try:
        result = llm.complete(
            system=EXECUTIVE_SYSTEM,
            user=payload_text,
            json_schema=_executive_summary_json_schema(),
            model=prose_model,
            surface="report",
        )
    except Exception as exc:  # noqa: BLE001 -- same "don't fail the whole report" contract as _headline
        if failure_reasons is not None:
            failure_reasons.append(f"the model call failed ({type(exc).__name__}: {exc})")
        return ExecutiveSummary(failure_reason=failure_reasons[-1] if failure_reasons else None)

    parsed = _ExecutiveSummaryLLM.model_validate(result.data)

    kept_this_week: list[str] = []
    for s in parsed.this_week:
        gated = _gate_sentence(
            conn,
            s,
            allowed_markers=allowed_markers,
            facts=facts,
            themes=themes,
            surface="executive_summary_this_week",
            grounded_texts=grounded_texts,
        )
        if gated:
            kept_this_week.append(gated)

    kept_attention: list[str] = []
    for s in parsed.attention:
        gated = _gate_sentence(
            conn,
            s,
            allowed_markers=allowed_markers,
            facts=facts,
            themes=themes,
            surface="executive_summary_attention",
            grounded_texts=grounded_texts,
        )
        if gated:
            kept_attention.append(gated)

    kept_actions: list[str] = []
    for s in parsed.suggested_actions:
        gated = _gate_sentence(
            conn,
            s,
            allowed_markers=allowed_markers,
            facts=facts,
            themes=themes,
            surface="executive_summary_suggested_actions",
            grounded_texts=grounded_texts,
        )
        if gated:
            kept_actions.append(gated)

    kept_by_account: list[AccountParagraph] = []
    for entry in parsed.by_account:
        if entry.account_slug not in account_slugs:
            _queue_executive_rejection(
                conn,
                surface="executive_summary_by_account",
                detail=(
                    f"account_slug not among this window's active accounts: {entry.account_slug!r}"
                ),
            )
            continue
        gated = _gate_sentence(
            conn,
            entry.paragraph,
            allowed_markers=allowed_markers,
            facts=facts,
            themes=themes,
            surface="executive_summary_by_account",
            grounded_texts=grounded_texts,
        )
        if gated:
            kept_by_account.append(
                AccountParagraph(account_slug=entry.account_slug, paragraph=gated)
            )

    total_lines = (
        (1 if kept_this_week else 0)
        + len(kept_attention)
        + len(kept_by_account)
        + len(kept_actions)
    )
    if total_lines > MAX_RENDERED_LINES:
        _queue_executive_rejection(
            conn,
            surface="executive_summary",
            detail=(
                f"assembled executive summary has {total_lines} lines, over the "
                f"{MAX_RENDERED_LINES}-line cap"
            ),
        )
        if failure_reasons is not None:
            failure_reasons.append(
                f"the executive summary was {total_lines} lines, over the "
                f"{MAX_RENDERED_LINES}-line cap -- rejected outright"
            )
        return ExecutiveSummary(failure_reason=failure_reasons[-1] if failure_reasons else None)

    if not (kept_this_week or kept_attention or kept_by_account or kept_actions):
        if failure_reasons is not None:
            parsed_anything = (
                parsed.this_week
                or parsed.attention
                or parsed.by_account
                or parsed.suggested_actions
            )
            if not parsed_anything:
                failure_reasons.append("the model returned nothing in any section")
            else:
                failure_reasons.append("every sentence in every section was rejected by a gate")
        return ExecutiveSummary(failure_reason=failure_reasons[-1] if failure_reasons else None)

    return ExecutiveSummary(
        this_week=kept_this_week,
        attention=kept_attention,
        by_account=kept_by_account,
        suggested_actions=kept_actions,
    )


def _source_slug_of(claim: dict) -> str:
    """The source slug out of `claim["citation"]` (`"^[source @ locator]"`)."""
    marker = claim["citation"]
    inner = marker[2:-1]  # strip the leading "^[" and trailing "]"
    source_slug, _at, _rest = inner.partition(" @ ")
    return source_slug.strip()
