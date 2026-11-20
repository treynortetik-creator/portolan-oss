"""Pydantic models shared across `portolan.report`
(`01-spec/surfacing-and-weekly-report.md`). Plain data; nothing here talks
to the database.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The values `themes.theme_trend` produces. "resolution proposed": a judge
# proposed a resolution and it awaits a human's confirm or reject.
Movement = Literal["new", "rising", "falling", "resolved", "resolution proposed"]


# ---------------------------------------------------------------------
# Feed 1: the diff (portolan/report/diff.py)
# ---------------------------------------------------------------------


class NoteChange(BaseModel):
    """One row of "notes created/updated/superseded in the window"."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    type: str
    title: str
    created: date
    updated: date
    status: str


class SupersededFact(BaseModel):
    """One claim superseded in the window (two clocks, spec sec 2).
    `new_source_slug`/`new_locator` cite the superseding claim."""

    model_config = ConfigDict(extra="forbid")

    note_slug: str
    old_claim_id: int
    new_claim_id: int
    old_text: str
    new_text: str
    new_source_slug: str
    new_locator: str
    superseded_at: datetime


class QuietAccounts(BaseModel):
    """Accounts with no claim mentioning them in 30/60/90 days. The lists
    nest: an account quiet at 90 days is also in the 60- and 30-day lists."""

    model_config = ConfigDict(extra="forbid")

    as_of: date
    quiet_30: list[str] = Field(default_factory=list)
    quiet_60: list[str] = Field(default_factory=list)
    quiet_90: list[str] = Field(default_factory=list)


class DiffFeed(BaseModel):
    """Feed 1, pure SQL, no model call."""

    model_config = ConfigDict(extra="forbid")

    since: date
    as_of: date
    notes_created: list[NoteChange] = Field(default_factory=list)
    notes_updated: list[NoteChange] = Field(default_factory=list)
    facts_superseded: list[SupersededFact] = Field(default_factory=list)
    quiet_accounts: QuietAccounts


# ---------------------------------------------------------------------
# Feed 2: themes (portolan/report/themes.py)
# ---------------------------------------------------------------------


class RepresentativeQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: int
    text: str
    source_slug: str
    locator: str
    # `None` when the claim resolves to no account (`home_account_slug`).
    account_slug: str | None = None


class SignalDetail(BaseModel):
    """One signal claim in full, for the report's expandable "N signals"
    list. Unlike `RepresentativeQuote` (a capped highlight), every signal
    behind the figure gets one."""

    model_config = ConfigDict(extra="forbid")

    claim_id: int
    fact_type: str
    account_slug: str | None = None
    claim_text: str
    quote: str | None = None
    source_slug: str
    locator: str
    note_slug: str


class PendingResolution(BaseModel):
    """One open (`status='proposed'`) resolution the judge proposed for a
    theme, with its cited evidence sentence. Silence-confirmed rows never
    appear: they have no evidence and the theme already reads `resolved`."""

    model_config = ConfigDict(extra="forbid")

    resolution_claim_id: int
    evidence: str
    source_slug: str
    locator: str


class ThemeSummary(BaseModel):
    """One theme note's current rollup stats."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    name: str
    signal_count: int
    account_slugs: list[str] = Field(default_factory=list)
    product_slugs: list[str] = Field(default_factory=list)
    linked_arr: float
    # Top-level accounts with no known ARR, excluded from `linked_arr`
    # rather than counted as $0.
    arr_unknown_accounts: int = 0
    # `account_slugs` holds leaf accounts; `linked_arr` sums top-level ones.
    # `{child: parent}` for each account rolled up, so both can be printed.
    rolled_up: dict[str, str] = Field(default_factory=dict)
    # `None` for a theme with no members (`MIN`/`MAX(added)` is NULL).
    first_seen: date | None
    last_seen: date | None
    movement: Movement
    # A retroactive peak on a past week (`themes.theme_peak`); `None` until
    # the arc has turned.
    peaked_week: date | None = None
    peaked_count: int | None = None
    quotes: list[RepresentativeQuote] = Field(default_factory=list)
    # At most one today (`uq_theme_resolutions_active_theme`), kept a list.
    pending_resolutions: list[PendingResolution] = Field(default_factory=list)
    # Every member claim, uncapped (unlike `quotes`).
    all_signals: list[SignalDetail] = Field(default_factory=list)


class ThemeReport(BaseModel):
    """What one `refresh_themes()` call did."""

    model_config = ConfigDict(extra="forbid")

    since: date
    as_of: date
    claims_considered: int = 0
    attached_existing: int = 0
    themes_created: int = 0
    outliers_remaining: int = 0
    llm_calls: int = 0
    # Resolution-judge health, so an outage is distinguishable from a quiet
    # week: attempts, attempts that raised, and whether
    # `THEME_JUDGE_MAX_CALLS` stopped the run early.
    judge_calls: int = 0
    judge_failures: int = 0
    judge_cap_hit: bool = False
    themes: list[ThemeSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Ranked priorities (portolan/report/priorities.py)
# ---------------------------------------------------------------------


class ThemePriority(BaseModel):
    """One theme's ranked-priority row for a `[since, as_of]` window
    (`priorities.compute_theme_priorities`).

    `account_slugs`/`account_count`: leaf accounts with a claim that joined
    the theme in the window. `arr`: summed over those accounts after
    roll-up (an account whose ancestor is also touched counts once);
    `arr_unknown_accounts` were excluded for unknown ARR. `regions`: sorted
    CRM regions of the leaf accounts. `rolled_up`: `{child: parent}`."""

    model_config = ConfigDict(extra="forbid")

    theme_slug: str
    theme_name: str
    movement: Movement
    account_slugs: list[str] = Field(default_factory=list)
    account_count: int
    arr: float
    arr_unknown_accounts: int = 0
    regions: list[str] = Field(default_factory=list)
    rolled_up: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------
# Waves (portolan/report/waves.py)
# ---------------------------------------------------------------------


class WaveRow(BaseModel):
    """One mechanical wave: claims sharing a `fact_type` and product family
    in the same calendar week, across `>= WAVE_MIN_ACCOUNTS` accounts after
    roll-up (`waves.compute_waves`). `account_slugs` lists the leaf accounts,
    before roll-up."""

    model_config = ConfigDict(extra="forbid")

    week_of: date
    fact_type: str
    product_family: str
    account_slugs: list[str] = Field(default_factory=list)
    linked_arr: float
    claim_ids: list[int] = Field(default_factory=list)
    # `{child: parent}` for each account rolled up.
    rolled_up: dict[str, str] = Field(default_factory=dict)
    # Top-level accounts with no known ARR, excluded from `linked_arr`.
    unknown_arr_count: int = 0
    # Citable quotes resolved in `compute_waves`, so rendering needs no `conn`.
    quotes: list[RepresentativeQuote] = Field(default_factory=list)
    # Every claim in the wave, uncapped.
    all_signals: list[SignalDetail] = Field(default_factory=list)


class WavesFeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    since: date
    as_of: date
    waves: list[WaveRow] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Feed 3: standing questions (portolan/report/questions.py)
# ---------------------------------------------------------------------


class StandingQuestionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: int
    reader: str
    question: str
    week: str
    answer_md: str
    citations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Public signals (portolan/report/public_signals.py)
# ---------------------------------------------------------------------


class PublicSignal(BaseModel):
    """One `public_signal` claim written by `portolan.signals` in the window."""

    model_config = ConfigDict(extra="forbid")

    claim_id: int
    account_slug: str | None
    category: str | None
    item_kind: str
    claim_text: str
    source_slug: str
    locator: str
    url: str
    recorded: date


# ---------------------------------------------------------------------
# Feed 4: outliers (portolan/report/outliers.py)
# ---------------------------------------------------------------------


class OutlierClaim(BaseModel):
    """A signal claim in the window attached to no theme
    (surfacing-and-weekly-report.md "4. Outliers")."""

    model_config = ConfigDict(extra="forbid")

    claim_id: int
    note_slug: str
    account_slug: str | None
    fact_type: str
    claim_text: str
    source_slug: str
    locator: str
    recorded: date


# ---------------------------------------------------------------------
# Account synthesis (portolan/report/synthesis.py)
# ---------------------------------------------------------------------


class SynthesisOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_slug: str
    # "kept_unchanged": passed every check but rendered identical to the
    # stored note, so nothing was written.
    status: Literal["rewritten", "kept_old_failed_check", "skipped_no_growth", "kept_unchanged"]
    reason: str | None = None


class SynthesisReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    since: date
    as_of: date
    outcomes: list[SynthesisOutcome] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Feed: actions (portolan/report/weekly.py, from the `actions` table)
# ---------------------------------------------------------------------


class ActionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    note_slug: str
    what: str
    owner: str | None
    # Stable per-owner key: two owners can share a display name, not a crm_id.
    owner_crm_id: str | None = None
    due: date | None
    status: str


class ActionsFeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    open: list[ActionRow] = Field(default_factory=list)
    overdue: list[ActionRow] = Field(default_factory=list)
    resolved_this_week: list[ActionRow] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Coverage (portolan/report/weekly.py, from coverage_ledger)
# ---------------------------------------------------------------------


class CoverageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    since: date
    as_of: date
    seen: int = 0
    extracted: int = 0
    partial: int = 0
    failed: int = 0
    sampled: int = 0
    filtered_by_reason: dict[str, int] = Field(default_factory=dict)
    sample_miss_count: int | None = None


# ---------------------------------------------------------------------
# Executive summary (portolan/report/executive.py): four named parts, so
# each can be gated and rendered on its own.
# ---------------------------------------------------------------------


class AccountParagraph(BaseModel):
    """One "By account" paragraph. `account_slug` is structured so the
    renderer, not the model, builds the `[[slug]]` heading link."""

    model_config = ConfigDict(extra="forbid")

    account_slug: str
    paragraph: str


class ExecutiveSummary(BaseModel):
    """The four parts of the executive summary after gating
    (`executive.build_executive_summary`). Lists may be shorter than the
    model's output, or empty on total failure with `failure_reason` set."""

    model_config = ConfigDict(extra="forbid")

    this_week: list[str] = Field(default_factory=list)
    attention: list[str] = Field(default_factory=list)
    by_account: list[AccountParagraph] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


# ---------------------------------------------------------------------
# The report itself (portolan/report/weekly.py)
# ---------------------------------------------------------------------


class ReportResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    week: str
    since: date
    as_of: date
    note_slug: str
    headline: list[str] = Field(default_factory=list)
    executive_summary: ExecutiveSummary = Field(default_factory=ExecutiveSummary)
    diff: DiffFeed
    themes: ThemeReport
    waves: WavesFeed
    priorities: list[ThemePriority] = Field(default_factory=list)
    standing_questions: list[StandingQuestionAnswer] = Field(default_factory=list)
    outliers: list[OutlierClaim] = Field(default_factory=list)
    actions: ActionsFeed
    coverage: CoverageSummary
    contact_changes: list[OutlierClaim] = Field(default_factory=list)
    public_signals: list[PublicSignal] = Field(default_factory=list)
    synthesis: SynthesisReport
    body_md: str = ""
