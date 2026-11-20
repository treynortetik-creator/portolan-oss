"""Turns one `review_queue` row into a human-readable card for `/review`.

Every `kind` the `review_queue_kind_check` constraint allows gets its own
presenter, answering in plain language: what Portolan was doing and where
it got stuck, the evidence, why it matters once resolved, and the choices.
`present_row` dispatches on `kind` (and, for `kind='other'`, on
`payload['reason']`); an unrecognised `other` reason gets a generic
sentence, with the raw payload only in the "Technical details" fold.

Pure functions: plain dicts in, `ReviewCard`/`Choice` dataclasses out -- no
database, HTTP or template. `portolan.web.routes.review` does the queries
and hands in the enriched row, so `tests/test_review_presenter.py` needs
no database.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from urllib.parse import quote as _urlquote

from markupsafe import Markup, escape

__all__ = [
    "Choice",
    "ReviewCard",
    "category_label_for",
    "highlight_numerals",
    "kind_counts",
    "present_row",
    "present_rows",
]

# ---------------------------------------------------------------------
# Vocabulary: every closed list this module turns into plain English.
# ---------------------------------------------------------------------

_SOURCE_TYPE_LABELS: dict[str, str] = {
    "call": "sales call",
    "email": "email",
    "ticket": "support ticket",
    "doc": "document",
    "crm": "CRM record",
    "chat": "chat conversation",
    "survey": "survey response",
    "rss": "news item",
    "edit": "human edit",
}

# extraction-schema-v0.md "Signal types, base set" (portolan.extract.schema
# .FactType), in plain words -- "the fact type in words ('a decision')".
_FACT_TYPE_LABELS: dict[str, str] = {
    "complaint": "a complaint",
    "request": "a request",
    "praise": "praise",
    "risk": "a risk",
    "competitor_mention": "a mention of a competitor",
    "commitment": "a commitment",
    "decision": "a decision",
    "contact_change": "a contact change",
    "pricing": "a pricing detail",
    "usage": "a usage note",
    "question": "a question",
    "other": "a fact",
}

# The small, stable set of plain-word category names the page's "counts
# by kind" strip and filter buttons group rows under. Coarser than the
# raw `kind`/`reason` pair on purpose -- a human filters by "what kind of
# thing is this", not by an internal bucket name.
_CATEGORY_UNKNOWN_ENTITY = "Unknown name"
_CATEGORY_AMBIGUOUS_ENTITY = "Ambiguous name"
_CATEGORY_FAILED_CONSOLIDATION = "Tidy-up failed"
_CATEGORY_THEME_RESOLUTION = "Proposed resolution"
_CATEGORY_REJECTED_DRAFT = "Rejected draft"
_CATEGORY_BAD_CRM_DATA = "Bad CRM data"
_CATEGORY_CONTRADICTION = "Possible contradiction"
_CATEGORY_QUOTE_UNSUPPORTED = "Unsupported quote"
_CATEGORY_OWNER_PROPOSAL = "Proposed owner"
_CATEGORY_ACTION_CARD = "Proposed action"
_CATEGORY_ACTION_DONE = "Proposed done"
_CATEGORY_OTHER = "Needs a look"

# `kind='other'` rows carry a sub-kind in `payload['reason']`. Only
# `category_label_for` reads the category maps below; each presenter sets its
# own `category_label`, so a new reason must be added to both places.
_DOWNGRADED_CONTRADICTION_REASON = (
    "contradicts verdict downgraded to extends: not both explicitly dated, or not strictly later"
)
_CONTRADICTION_REASONS = frozenset(
    {
        "contradicts verdict blocked: claim was already superseded",
        _DOWNGRADED_CONTRADICTION_REASON,
    }
)
_CATEGORY_BY_OTHER_REASON: dict[str, str] = {
    "numeral_not_supplied": _CATEGORY_REJECTED_DRAFT,
    "resolved_language_without_a_resolved_theme": _CATEGORY_REJECTED_DRAFT,
    "failed account synthesis": _CATEGORY_REJECTED_DRAFT,
    "arr_not_parseable": _CATEGORY_BAD_CRM_DATA,
    "no_home_note": _CATEGORY_OTHER,
}
_CATEGORY_BY_KIND: dict[str, str] = {
    "unknown_entity": _CATEGORY_UNKNOWN_ENTITY,
    "ambiguous_entity": _CATEGORY_AMBIGUOUS_ENTITY,
    "failed_consolidation": _CATEGORY_FAILED_CONSOLIDATION,
    "theme_resolution": _CATEGORY_THEME_RESOLUTION,
    "quote_unsupported": _CATEGORY_QUOTE_UNSUPPORTED,
    "owner_proposal": _CATEGORY_OWNER_PROPOSAL,
    "action_card": _CATEGORY_ACTION_CARD,
    "action_done": _CATEGORY_ACTION_DONE,
}

_SURFACE_LABELS: dict[str, str] = {
    "headline": "this week's report headline",
    "account_synthesis": "an account's summary",
    "theme_synthesis": "a theme's summary",
    "theme_name": "a theme's name",
    "standing_question": "the answer to a standing question",
    "executive_summary_this_week": "the report's “This week” summary",
    "executive_summary_attention": "the report's “What needs attention” list",
    "executive_summary_by_account": "the report's “By account” section",
    "executive_summary_suggested_actions": "the report's “Suggested actions” list",
}

_IGNORE_CONFIRM = "Ignore this? This can’t be undone."


@dataclass
class Choice:
    """One button on a card. `action` names the route it posts to:
    `/review/{id}/resolve`, `/review/{id}/dismiss` or `/review/{id}/add-alias`."""

    action: str  # "alias" | "resolve" | "dismiss"
    label: str
    description: str
    confirm: str | None = None


@dataclass
class ReviewCard:
    """Everything one review-queue row's card needs to render, already in
    plain English. Every field a template needs is named; nothing here is
    a raw id, slug, or payload key meant for a human's eyes -- those live
    in `technical` instead (`[(label, value), ...]`, rendered inside the
    page's collapsed "Technical details" fold)."""

    row_id: int
    kind: str
    category_label: str
    created: datetime | None
    headline: str
    why_it_matters: str
    choices: list[Choice]
    technical: list[tuple[str, str]]

    # Evidence block: the fact or claim under review and where it came from.
    claim_text: str | None = None
    quote: str | None = None
    context_before: str | None = None
    context_after: str | None = None
    fact_type_label: str | None = None
    source_label: str | None = None
    open_url: str | None = None
    open_label: str | None = None
    # Set when an entity row has no saved claim and the backfill found none.
    evidence_unavailable_note: str | None = None

    # unknown_entity / ambiguous_entity: the alias-picker form is shown.
    show_alias_picker: bool = False
    mention: str | None = None

    # ambiguous_entity only.
    candidates: list[dict[str, str | None]] = field(default_factory=list)

    # theme_resolution only.
    theme_name: str | None = None
    theme_slug: str | None = None
    account_slug: str | None = None

    # failed_consolidation and several 'other' reasons: the note concerned.
    note_slug: str | None = None
    note_title: str | None = None
    reasons: list[str] = field(default_factory=list)

    # 'other' rejected drafts: numeral_not_supplied sets all four;
    # resolved_language_without_a_resolved_theme sets text and surface.
    rejected_text: str | None = None
    offending_numerals: list[str] = field(default_factory=list)
    allowed_figures: list[tuple[str, str]] = field(default_factory=list)
    surface_label: str | None = None

    # crm_id: arr_not_parseable and owner_proposal. detail: arr_not_parseable
    # and failed account synthesis.
    crm_id: str | None = None
    detail: str | None = None

    # owner_proposal only: ranked candidates, each {crm_id, name,
    # source_count, last_seen, commitments} (not ambiguous_entity's shape).
    owner_candidates: list[dict[str, Any]] = field(default_factory=list)

    # action_card (01-spec/actions-and-owners-2026-09-19.md "Phase 2, revised
    # contract", "Review page"). action_done also sets what, owner name,
    # account name/slug and due. Evidence, findings, inference and
    # confidence refs arrive already resolved (`_enrich_action_card_row`).
    action_verdict: str | None = None
    action_what: str | None = None
    action_owner_name: str | None = None
    action_owner_crm_id: str | None = None
    action_account_name: str | None = None
    action_account_slug: str | None = None
    action_theme_name: str | None = None
    action_theme_slug: str | None = None
    action_due: str | None = None
    action_due_rule: str | None = None
    action_confidence: str | None = None
    action_evidence: list[dict[str, Any]] = field(default_factory=list)
    action_findings: list[dict[str, Any]] = field(default_factory=list)
    action_inference: list[dict[str, Any]] = field(default_factory=list)
    action_confidence_refs: dict[str, Any] = field(default_factory=dict)
    action_trail: dict[str, Any] = field(default_factory=dict)
    action_owner_choices: list[dict[str, Any]] = field(default_factory=list)

    # action_done only: the open action a later claim looks like it answers.
    # The claim itself uses the evidence fields above.
    done_action_slug: str | None = None
    done_action_date: str | None = None
    done_action_status: str | None = None
    done_finding_text: str | None = None


# ---------------------------------------------------------------------
# Small formatting helpers shared by more than one presenter.
# ---------------------------------------------------------------------


def _fmt_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    return text or None


def _participants_text(participants: list[dict[str, Any]]) -> str | None:
    names = [p.get("name") for p in participants if p.get("name")]
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]}, {names[1]}, and {len(names) - 2} more"


def _join_or(items: list[str]) -> str | None:
    items = [i for i in items if i]
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def _source_label(source: dict[str, Any] | None) -> str | None:
    """A plain-language source label, never the slug. `sources` has no
    title column, so it is built from type, the matched account name (else
    the participants) and date."""
    if not source:
        return None
    source_type = source.get("source_type")
    kind_word = _SOURCE_TYPE_LABELS.get(source_type, source_type or "record")
    who = source.get("account_name") or _participants_text(source.get("participants") or [])
    date_str = _fmt_date(source.get("source_date"))
    bits = [f"the {kind_word}"]
    if who:
        bits.append(f"with {who}")
    if date_str:
        bits.append(f"on {date_str}")
    return " ".join(bits)


_EVIDENCE_UNAVAILABLE_NOTE = (
    "Portolan no longer has the exact sentence for this mention; open the original "
    "to see it in context."
)


def _apply_evidence_backfill(
    row: dict[str, Any],
    *,
    claim: str | None,
    quote: str | None,
    fact_type: str | None,
    locator: str | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Fill missing evidence from `row['evidence_backfill']` when the row
    saved no `claim` (`Fact.claim` is required, so `None` means not saved).
    An absent and a `None` backfill both mean nothing to show.

    Returns `(claim, quote, fact_type, locator, evidence_unavailable_note)`;
    the note is set exactly when `claim` is still `None`."""
    if claim is not None:
        return claim, quote, fact_type, locator, None
    backfill = row.get("evidence_backfill")
    if backfill:
        claim = backfill.get("claim")
        quote = quote or backfill.get("quote")
        fact_type = fact_type or backfill.get("fact_type")
        locator = locator or backfill.get("locator")
    note = _EVIDENCE_UNAVAILABLE_NOTE if claim is None else None
    return claim, quote, fact_type, locator, note


def _open_original(row: dict[str, Any], locator: str | None) -> tuple[str | None, str | None]:
    source_slug = row.get("source_slug")
    if not source_slug:
        return None, None
    url = f"/originals/{source_slug}"
    if locator:
        url += f"?at={_urlquote(locator)}"
    return url, "Open the original →"


def _technical(row: dict[str, Any], extra: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = [
        ("review row id", str(row.get("id"))),
        ("kind", str(row.get("kind"))),
    ]
    if row.get("source_slug"):
        items.append(("source", str(row["source_slug"])))
    for key, value in (extra or {}).items():
        if value is None or value == "":
            continue
        items.append((key, _stringify(value)))
    return items


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _basic_choices(*, ignore_label: str = "Ignore this") -> list[Choice]:
    return [
        Choice(
            action="resolve",
            label="Handled, nothing to change",
            description="Mark this as looked at. Nothing else on the graph changes.",
        ),
        Choice(
            action="dismiss",
            label=ignore_label,
            description="Drop it. Portolan won't ask about this again.",
            confirm=_IGNORE_CONFIRM,
        ),
    ]


def _mention_choices(mention: str) -> list[Choice]:
    return [
        Choice(
            action="alias",
            label="This is another name for…",
            description=(
                f"Pick which record “{mention}” actually refers to. Every future "
                "mention of it resolves the same way automatically."
            ),
        ),
        Choice(
            action="resolve",
            label="Handled, nothing to change",
            description="Mark this as looked at without linking it to anything.",
        ),
        Choice(
            action="dismiss",
            label="Ignore this mention",
            description="Drop it. Portolan won't ask about this exact mention again.",
            confirm="Ignore this mention? This can’t be undone.",
        ),
    ]


# ---------------------------------------------------------------------
# Per-kind presenters.
# ---------------------------------------------------------------------


def _more_sources_sentence(payload: dict[str, Any], mention: str) -> str:
    """`registry.park_mention` keeps one open row per name and adds each
    further source to `payload.occurrences`; one answer covers them all."""
    more = len(payload.get("occurrences") or [])
    if not more:
        return ""
    sources = "source" if more == 1 else "sources"
    return f" “{mention}” also came up in {more} other {sources}; one answer covers all of them."


def _occurrence_slugs(payload: dict[str, Any]) -> str | None:
    slugs = [o.get("source_slug") for o in payload.get("occurrences") or [] if o.get("source_slug")]
    return ", ".join(slugs) if slugs else None


def _present_unknown_entity(row: dict[str, Any]) -> ReviewCard:
    payload = row.get("payload") or {}
    mention = payload.get("mention") or "(unnamed)"
    context = payload.get("context") or {}
    claim, quote, fact_type, locator, evidence_unavailable_note = _apply_evidence_backfill(
        row,
        claim=context.get("claim"),
        quote=context.get("quote"),
        fact_type=context.get("fact_type"),
        locator=context.get("locator"),
    )
    fact_type_label = _FACT_TYPE_LABELS.get(fact_type) if fact_type else None

    source_label = _source_label(row.get("source"))
    open_url, open_label = _open_original(row, locator)
    snippet = row.get("context_snippet") or {}

    if source_label:
        headline = (
            f"While reading {source_label}, Portolan found the name “{mention}” "
            "but couldn't tell which account, contact, or product it refers to."
        )
    else:
        headline = (
            f"Portolan found the name “{mention}” but couldn't tell which account, "
            "contact, or product it refers to."
        )
    headline += _more_sources_sentence(payload, mention)

    why_it_matters = (
        f"Once you tell Portolan who or what “{mention}” is, this fact is filed "
        "under that record and shows up in its reports. Until then it's parked here, "
        "unattached to anything."
    )

    technical = _technical(
        row,
        extra={
            "fact id": context.get("fact_id"),
            "locator": locator,
            "mention": mention,
            "also named in": _occurrence_slugs(payload),
        },
    )

    return ReviewCard(
        row_id=row["id"],
        kind="unknown_entity",
        category_label=_CATEGORY_UNKNOWN_ENTITY,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_mention_choices(mention),
        technical=technical,
        claim_text=claim,
        quote=quote,
        context_before=snippet.get("before"),
        context_after=snippet.get("after"),
        fact_type_label=fact_type_label,
        source_label=source_label,
        open_url=open_url,
        open_label=open_label,
        show_alias_picker=True,
        mention=mention,
        evidence_unavailable_note=evidence_unavailable_note,
    )


def _present_ambiguous_entity(
    row: dict[str, Any], entities_by_crm_id: dict[str, dict[str, Any]]
) -> ReviewCard:
    payload = row.get("payload") or {}
    mention = payload.get("mention") or "(unnamed)"
    candidate_ids = payload.get("candidates") or []
    reason = payload.get("reason")

    candidates: list[dict[str, str | None]] = []
    for crm_id in candidate_ids:
        info = entities_by_crm_id.get(crm_id)
        candidates.append(
            {
                "crm_id": crm_id,
                "name": (info or {}).get("canonical_name") or crm_id,
                "kind": (info or {}).get("kind"),
            }
        )
    candidates_text = _join_or([c["name"] for c in candidates])

    # The CRM import's same-name check writes these rows with no source; a
    # document mention always has one. `reason` is display copy, not a code.
    is_crm_import = not row.get("source_slug")

    if is_crm_import:
        what = f" ({candidates_text})" if candidates_text else ""
        headline = (
            f"While importing the CRM, Portolan found two records both named "
            f"“{mention}”{what} with different email domains, so it kept both "
            "rather than assuming they're the same company."
        )
        why_it_matters = (
            "If they really are the same company, mark one as another name for the other and "
            "Portolan merges their history together. If they're genuinely two different "
            "companies that happen to share a name, there's nothing to do."
        )
        claim = quote = fact_type_label = source_label = open_url = open_label = None
        context_before = context_after = evidence_unavailable_note = None
    else:
        claim, quote, fact_type, locator, evidence_unavailable_note = _apply_evidence_backfill(
            row,
            claim=payload.get("claim"),
            quote=payload.get("quote"),
            fact_type=payload.get("fact_type"),
            locator=payload.get("locator"),
        )
        fact_type_label = _FACT_TYPE_LABELS.get(fact_type) if fact_type else None
        source_label = _source_label(row.get("source"))
        open_url, open_label = _open_original(row, locator)
        snippet = row.get("context_snippet") or {}
        context_before = snippet.get("before")
        context_after = snippet.get("after")

        where = f"While reading {source_label}, " if source_label else ""
        headline = (
            f"{where}Portolan found the name “{mention}” but it could mean "
            f"{candidates_text or 'more than one thing'}, and couldn't tell which one."
        ) + _more_sources_sentence(payload, mention)
        why_it_matters = (
            f"Once you tell Portolan which one “{mention}” means, this fact is filed "
            "under the right record and shows up in its reports."
        )

    technical = _technical(
        row,
        extra={"reason": reason, "mention": mention, "also named in": _occurrence_slugs(payload)},
    )

    return ReviewCard(
        row_id=row["id"],
        kind="ambiguous_entity",
        category_label=_CATEGORY_AMBIGUOUS_ENTITY,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_mention_choices(mention),
        technical=technical,
        claim_text=claim,
        quote=quote,
        context_before=context_before,
        context_after=context_after,
        fact_type_label=fact_type_label,
        source_label=source_label,
        open_url=open_url,
        open_label=open_label,
        show_alias_picker=True,
        mention=mention,
        candidates=candidates,
        evidence_unavailable_note=evidence_unavailable_note,
    )


def _present_failed_consolidation(row: dict[str, Any]) -> ReviewCard:
    payload = row.get("payload") or {}
    note_slug = payload.get("note_slug")
    note_title = row.get("note_title") or note_slug
    reasons = list(payload.get("reasons") or [])

    if note_title:
        headline = (
            f"While tidying up the note “{note_title}” (folding its older facts "
            "into a shorter summary), Portolan's rewrite failed its own quality check and "
            "was thrown out -- nothing on the note changed."
        )
    else:
        headline = (
            "Portolan tried to tidy up a note (folding its older facts into a shorter "
            "summary), but the rewrite failed its own quality check and was thrown out."
        )

    why_it_matters = (
        "The note stays exactly as it was until this is resolved; new facts keep being "
        "added underneath, it just won't get any shorter for now."
    )

    technical = _technical(row, extra={"note_slug": note_slug})

    return ReviewCard(
        row_id=row["id"],
        kind="failed_consolidation",
        category_label=_CATEGORY_FAILED_CONSOLIDATION,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_basic_choices(),
        technical=technical,
        note_slug=note_slug,
        note_title=note_title,
        reasons=reasons,
        open_url=f"/notes/{note_slug}" if note_slug else None,
        open_label="Open the note →" if note_slug else None,
    )


def _present_theme_resolution(row: dict[str, Any]) -> ReviewCard:
    payload = row.get("payload") or {}
    proposal = row.get("proposal")
    theme_slug = payload.get("theme_slug")

    snippet = row.get("context_snippet") or {}

    if proposal is None:
        headline = (
            "Portolan proposed that a theme is now resolved, but the theme or the claim it "
            "pointed at no longer exists, so there's nothing left to show."
        )
        why_it_matters = "Dismiss this -- there's nothing here to confirm or reject."
        theme_name = theme_slug
        account_slug = None
        world_date = None
        claim = quote = fact_type_label = source_label = open_url = open_label = None
    else:
        theme_name = proposal.get("theme_name") or theme_slug
        account_slug = proposal.get("account_slug")
        world_date = _fmt_date(proposal.get("world_date"))
        claim = proposal.get("claim_text")
        quote = payload.get("evidence") or proposal.get("quote")
        fact_type = proposal.get("fact_type")
        fact_type_label = _FACT_TYPE_LABELS.get(fact_type) if fact_type else None
        source_slug = proposal.get("source_slug")
        locator = proposal.get("locator")
        source_label = None
        open_url, open_label = _open_original({"source_slug": source_slug}, locator)

        on_date = f" on {world_date}" if world_date else ""
        headline = (
            f"Portolan noticed a new claim{on_date} that looks like it resolves the theme "
            f"“{theme_name}”, and wants to mark that theme resolved."
        )
        why_it_matters = (
            "Confirming closes this theme as resolved in every future report. Rejecting "
            "keeps the theme open, and Portolan won't propose this exact claim again."
        )

    choices = [
        Choice(
            action="resolve",
            label="Confirm resolution",
            description=(
                "Yes -- this claim genuinely resolves the theme. It reads resolved from now on."
            ),
        ),
        Choice(
            action="dismiss",
            label="Reject",
            description="No -- this doesn't resolve it. The theme stays open.",
            confirm=(
                "Reject this resolution? This exact claim will not be judged again for this theme."
            ),
        ),
    ]

    technical = _technical(
        row,
        extra={
            "theme_slug": theme_slug,
            "resolution_claim_id": payload.get("resolution_claim_id"),
            "resolved_week": payload.get("resolved_week"),
        },
    )

    return ReviewCard(
        row_id=row["id"],
        kind="theme_resolution",
        category_label=_CATEGORY_THEME_RESOLUTION,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=choices,
        technical=technical,
        claim_text=claim,
        quote=quote,
        context_before=snippet.get("before"),
        context_after=snippet.get("after"),
        fact_type_label=fact_type_label,
        source_label=source_label,
        open_url=open_url,
        open_label=open_label,
        theme_name=theme_name,
        theme_slug=theme_slug,
        account_slug=account_slug,
    )


def _quote_unsupported_choices() -> list[Choice]:
    return [
        Choice(
            action="resolve",
            label="Keep as claim",
            description=(
                "File it anyway, if you've checked the original and the quote is a fair "
                "(if imperfect) rendering of what it really says. Portolan re-runs this "
                "one fact through the normal merge, exactly as if its quote had passed."
            ),
        ),
        Choice(
            action="dismiss",
            label="Dismiss",
            description="Drop it. This fact never becomes a claim.",
            confirm=_IGNORE_CONFIRM,
        ),
    ]


def _present_quote_unsupported(row: dict[str, Any]) -> ReviewCard:
    """A fact the quote gate held back because its quote was not found in
    the source (`portolan.extract.quote_gate.quote_is_supported`).
    `payload['fact']` is the full fact, so no backfill is needed."""
    payload = row.get("payload") or {}
    fact = payload.get("fact") or {}
    claim = fact.get("claim")
    quote = fact.get("quote")
    locator = fact.get("locator")
    fact_type = fact.get("type")
    fact_type_label = _FACT_TYPE_LABELS.get(fact_type) if fact_type else None

    source_label = _source_label(row.get("source"))
    open_url, open_label = _open_original(row, locator)
    snippet = row.get("context_snippet") or {}

    if source_label:
        headline = (
            f"While reading {source_label}, the model quoted words that are not in the source."
        )
    else:
        headline = "The model quoted words that are not in the source."

    why_it_matters = (
        "Portolan only writes a claim from a quote it can actually find in the source it "
        "cites -- this one's words don't appear there, verbatim or close to it, so the fact "
        "was held back rather than filed. Dismiss it if the quote looks made up; keep it "
        "only once you've checked the original yourself and the quote is a fair account of "
        "what it says."
    )

    technical = _technical(row, extra={"fact id": fact.get("id"), "locator": locator})

    return ReviewCard(
        row_id=row["id"],
        kind="quote_unsupported",
        category_label=_CATEGORY_QUOTE_UNSUPPORTED,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_quote_unsupported_choices(),
        technical=technical,
        claim_text=claim,
        quote=quote,
        context_before=snippet.get("before"),
        context_after=snippet.get("after"),
        fact_type_label=fact_type_label,
        source_label=source_label,
        open_url=open_url,
        open_label=open_label,
    )


def _owner_proposal_choices() -> list[Choice]:
    return [
        Choice(
            action="resolve",
            label="Confirm owner",
            description=(
                "Set this account's owner to the top-ranked candidate below. The account "
                "note, Atlas, the Actions page and the weekly report all read the owner from "
                "this one place."
            ),
        ),
        Choice(
            action="dismiss",
            label="Dismiss",
            description="Drop this proposal. The account's owner is left unset.",
            confirm=_IGNORE_CONFIRM,
        ),
    ]


def _present_owner_proposal(row: dict[str, Any]) -> ReviewCard:
    """One proposed owner per account (`python -m portolan.registry
    propose-owners`): `payload = {account_crm_id, account_name, candidates:
    [{crm_id, name, source_count, last_seen, commitments}, ...]}`, ranked
    best first. Confirm always takes the top candidate; there is no picker."""
    payload = row.get("payload") or {}
    account_crm_id = payload.get("account_crm_id")
    account_name = payload.get("account_name") or account_crm_id or "(unknown account)"
    candidates = list(payload.get("candidates") or [])
    top = candidates[0] if candidates else None

    if top:
        source_count = top.get("source_count") or 0
        bits = [f"{source_count} source{'s' if source_count != 1 else ''}"]
        if top.get("last_seen"):
            bits.append(f"last seen {top['last_seen']}")
        commitments = top.get("commitments") or 0
        if commitments:
            bits.append(f"{commitments} commitment{'s' if commitments != 1 else ''}")
        headline = (
            f"Portolan's evidence points to {top.get('name') or top.get('crm_id')} as the "
            f"owner of “{account_name}” ({', '.join(bits)})."
        )
    else:
        headline = (
            f"Portolan flagged a possible owner for “{account_name}”, but no candidate "
            "evidence was recorded with it."
        )

    why_it_matters = (
        "Confirming sets registry_entities.owner_entity_id for this account -- a re-ingest "
        "or merge never clobbers it. Dismiss leaves the account unowned; Portolan will "
        "propose again on the next run if the evidence still points the same way."
    )

    technical = _technical(row, extra={"account_crm_id": account_crm_id})

    return ReviewCard(
        row_id=row["id"],
        kind="owner_proposal",
        category_label=_CATEGORY_OWNER_PROPOSAL,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_owner_proposal_choices(),
        technical=technical,
        crm_id=account_crm_id,
        owner_candidates=candidates,
    )


def _action_card_choices() -> list[Choice]:
    return [
        Choice(
            action="resolve",
            label="Confirm",
            description=(
                "Create this action: owner, due date, status open, source and locator "
                "from its first evidence claim. Edit the owner or due date below first "
                "if the researcher got either wrong."
            ),
        ),
        Choice(
            action="dismiss",
            label="Dismiss",
            description="Drop this proposed action. Nothing is created.",
            confirm=_IGNORE_CONFIRM,
        ),
    ]


_VERDICT_LABEL = {
    "action": "a possible action",
    "no_action": "no action needed",
    "trigger_stale": "a stale trigger",
}


def _present_action_card(row: dict[str, Any]) -> ReviewCard:
    """The action researcher's card (01-spec/actions-and-owners-2026-09-19.md
    "Phase 2, revised contract"). Arranges `row['action_context']`, built by
    the route; without it the card falls back to the payload's own fields
    and empty lists."""
    payload = row.get("payload") or {}
    card = payload.get("card") or {}
    ctx = row.get("action_context") or {}

    verdict = card.get("verdict") or "action"
    account_name = ctx.get("account_name") or card.get("account") or "(unknown account)"
    owner_name = ctx.get("owner_name") or card.get("owner") or "(unassigned)"
    raw_due = card.get("due")
    due = _fmt_date(raw_due) if raw_due else None
    due_txt = f" by {due}" if due else ""

    headline = (
        f"Portolan researched {_VERDICT_LABEL.get(verdict, verdict)} for "
        f"{account_name}{due_txt}: {card.get('what', '')}"
    )

    if verdict == "action":
        why_it_matters = (
            "Confirming creates a real, open action -- owner, due date, and the source it "
            "cites all come from the card below. Dismiss records why and creates nothing; "
            "the underlying candidate may be researched again on a future run."
        )
    else:
        why_it_matters = (
            "The researcher concluded this candidate doesn't need a new action (see the "
            "evidence and findings below for why). Confirming records that conclusion; "
            "Dismiss just clears the row -- either way nothing new is created."
        )

    technical = _technical(
        row,
        extra={
            "candidate_slug": card.get("candidate_slug"),
            "brief_version": card.get("brief_version"),
            "run_id": card.get("run_id"),
        },
    )

    return ReviewCard(
        row_id=row["id"],
        kind="action_card",
        category_label=_CATEGORY_ACTION_CARD,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_action_card_choices(),
        technical=technical,
        action_verdict=verdict,
        action_what=card.get("what"),
        action_owner_name=owner_name,
        action_owner_crm_id=card.get("owner"),
        action_account_name=account_name,
        action_account_slug=card.get("account"),
        action_theme_name=ctx.get("theme_name"),
        action_theme_slug=card.get("theme"),
        action_due=due,
        action_due_rule=card.get("due_rule"),
        action_confidence=card.get("confidence"),
        action_evidence=ctx.get("evidence") or [],
        action_findings=ctx.get("findings") or [],
        action_inference=ctx.get("inference") or [],
        action_confidence_refs=ctx.get("confidence_refs") or {},
        action_trail=ctx.get("trail") or {},
        action_owner_choices=ctx.get("owner_choices") or [],
    )


def _action_done_choices() -> list[Choice]:
    return [
        Choice(
            action="resolve",
            label="Confirm done",
            description=(
                "Mark the action done, signed with your name. The claim above stays on the "
                "action as the reason it closed."
            ),
        ),
        Choice(
            action="dismiss",
            label="Dismiss",
            description=(
                "This claim doesn't answer the action. The action stays open, and Portolan "
                "won't propose this claim for it again."
            ),
            confirm="Dismiss this? Portolan won't propose this claim for this action again.",
        ),
    ]


def _present_action_done(row: dict[str, Any]) -> ReviewCard:
    """`portolan.report.action_done`'s proposal: a later claim by someone on
    our side that a judge read as answering an open action. `row['done_
    context']` supplies account and owner names and the action's current
    status and due date; without it the card falls back to the payload."""
    payload = row.get("payload") or {}
    ctx = row.get("done_context") or {}
    evidence = payload.get("evidence") or {}
    what = payload.get("what") or "(no description)"
    account_name = ctx.get("account_name") or payload.get("account") or "(unknown account)"
    speaker = payload.get("speaker") or "Someone on our side"
    claim_date = _fmt_date(payload.get("claim_date"))
    on_date = f" on {claim_date}" if claim_date else ""
    fact_type = payload.get("fact_type")

    headline = (
        f"{speaker} recorded something{on_date} that looks like it answers the open action "
        f"“{what}” for {account_name}, so Portolan proposes marking it done."
    )
    why_it_matters = (
        "Confirming marks the action done on the Actions page, signed with your name, and "
        "keeps this claim on the action as the reason it closed. Dismiss leaves the action "
        "open."
    )
    locator = evidence.get("locator")
    open_url, open_label = _open_original(row, locator)
    snippet = row.get("context_snippet") or {}
    technical = _technical(
        row,
        extra={
            "action id": payload.get("action_id"),
            "claim id": payload.get("claim_id"),
            "finding id": payload.get("finding_id"),
            "run id": payload.get("run_id"),
            "locator": locator,
        },
    )
    return ReviewCard(
        row_id=row["id"],
        kind="action_done",
        category_label=_CATEGORY_ACTION_DONE,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_action_done_choices(),
        technical=technical,
        claim_text=evidence.get("text"),
        quote=evidence.get("quote"),
        context_before=snippet.get("before"),
        context_after=snippet.get("after"),
        fact_type_label=_FACT_TYPE_LABELS.get(fact_type) if fact_type else None,
        source_label=_source_label(row.get("source")),
        open_url=open_url,
        open_label=open_label,
        action_what=what,
        action_owner_name=ctx.get("owner_name") or payload.get("owner_crm_id"),
        action_account_name=account_name,
        action_account_slug=payload.get("account"),
        action_due=_fmt_date(ctx.get("due")),
        done_action_slug=payload.get("action_slug"),
        done_action_date=_fmt_date(payload.get("action_date")),
        done_action_status=ctx.get("status"),
        done_finding_text=ctx.get("finding_text"),
    )


def highlight_numerals(text: str | None, violations: list[str]) -> Markup:
    """`text` HTML-escaped, with each exact string in `violations` wrapped
    in `<mark>`, as `Markup` safe for `|safe` (the `highlight_numerals`
    Jinja filter)."""
    if not text:
        return Markup("")
    words = sorted({v for v in violations if v}, key=len, reverse=True)
    if not words:
        return Markup(escape(text))
    pattern = re.compile("|".join(re.escape(w) for w in words))
    pieces: list[Markup] = []
    last = 0
    for match in pattern.finditer(text):
        pieces.append(Markup(escape(text[last : match.start()])))
        pieces.append(Markup("<mark>") + Markup(escape(match.group(0))) + Markup("</mark>"))
        last = match.end()
    pieces.append(Markup(escape(text[last:])))
    return Markup("").join(pieces)


def _present_numeral_rejection(row: dict[str, Any], payload: dict[str, Any]) -> ReviewCard:
    surface = payload.get("surface")
    surface_label = _SURFACE_LABELS.get(surface, surface or "some report text")
    note_slug = payload.get("note_slug")
    text = payload.get("text")
    violations = list(payload.get("violations") or [])
    allowed_figures_raw = payload.get("allowed_figures")

    headline = (
        f"Portolan drafted {surface_label}, but it used a number that wasn't in the figures "
        "it was allowed to use, so the whole thing was thrown out rather than shipped with a "
        "made-up number."
    )
    why_it_matters = (
        "Nothing wrong was shown to anyone -- the draft never went out. Portolan will draft "
        "this again next run; there's nothing to fix here besides looking it over."
    )

    allowed_figures: list[tuple[str, str]] = []
    if allowed_figures_raw:
        for line in str(allowed_figures_raw).splitlines():
            line = line.strip().lstrip("-").strip()
            if not line or ":" not in line:
                continue
            name, _, value = line.partition(":")
            allowed_figures.append((name.strip().replace("_", " "), value.strip()))

    technical = _technical(row, extra={"surface": surface, "note_slug": note_slug})

    return ReviewCard(
        row_id=row["id"],
        kind="other",
        category_label=_CATEGORY_REJECTED_DRAFT,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_basic_choices(),
        technical=technical,
        rejected_text=text,
        offending_numerals=violations,
        allowed_figures=allowed_figures,
        surface_label=surface_label,
        note_slug=note_slug,
        open_url=f"/notes/{note_slug}" if note_slug else None,
        open_label="Open the note →" if note_slug else None,
    )


def _present_resolved_language_rejection(
    row: dict[str, Any], payload: dict[str, Any]
) -> ReviewCard:
    sentence = payload.get("detail")
    surface = payload.get("surface") or "headline"
    surface_label = _SURFACE_LABELS.get(surface, "some report text")
    headline = (
        f"Portolan drafted a sentence for {surface_label} that talks about a problem being "
        "resolved, but no theme in this report actually reads resolved (or the sentence "
        "named the wrong one) -- so the sentence was dropped."
    )
    why_it_matters = (
        "Nothing wrong was shown to anyone -- the sentence never went out. Portolan will "
        "draft this text again next run."
    )
    technical = _technical(row, extra={"surface": surface})
    return ReviewCard(
        row_id=row["id"],
        kind="other",
        category_label=_CATEGORY_REJECTED_DRAFT,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_basic_choices(),
        technical=technical,
        rejected_text=sentence,
        surface_label=surface_label,
    )


def _present_failed_synthesis(row: dict[str, Any], payload: dict[str, Any]) -> ReviewCard:
    note_slug = payload.get("note_slug")
    note_title = row.get("note_title") or note_slug
    detail = payload.get("detail")
    subject = f"the account “{note_title}”" if note_title else "an account"
    headline = (
        f"While rewriting the summary for {subject}, Portolan's draft failed its own quality "
        f"check ({detail}), so the old summary was kept instead."
    )
    why_it_matters = "The account's summary is unchanged for now. Portolan will try again next run."
    technical = _technical(row, extra={"note_slug": note_slug})
    return ReviewCard(
        row_id=row["id"],
        kind="other",
        category_label=_CATEGORY_REJECTED_DRAFT,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_basic_choices(),
        technical=technical,
        note_slug=note_slug,
        note_title=note_title,
        detail=detail,
        open_url=f"/notes/{note_slug}" if note_slug else None,
        open_label="Open the note →" if note_slug else None,
    )


def _present_arr_not_parseable(row: dict[str, Any], payload: dict[str, Any]) -> ReviewCard:
    crm_id = payload.get("crm_id")
    detail = payload.get("detail")
    headline = (
        f"While reading the CRM export, Portolan found a revenue (ARR) figure for account "
        f"{crm_id} that isn't a valid number. It's being treated as unknown, not zero, until "
        "this is fixed in the CRM."
    )
    why_it_matters = (
        "Reports and account pages will show this account's revenue as unknown, not $0, "
        "until the CRM export is corrected and re-imported."
    )
    technical = _technical(row, extra={"crm_id": crm_id})
    return ReviewCard(
        row_id=row["id"],
        kind="other",
        category_label=_CATEGORY_BAD_CRM_DATA,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_basic_choices(),
        technical=technical,
        crm_id=crm_id,
        detail=detail,
    )


def _present_contradiction(
    row: dict[str, Any], payload: dict[str, Any], *, downgraded: bool
) -> ReviewCard:
    note_slug = payload.get("note_slug")
    note_title = row.get("note_title") or note_slug
    subject = f"the note “{note_title}”" if note_title else "a note"
    if downgraded:
        headline = (
            f"Portolan noticed two facts on {subject} that might contradict each other, but "
            "couldn't prove it (both need clear, ordered dates), so it kept both as still "
            "true instead of retiring the older one."
        )
    else:
        headline = (
            f"Portolan found a new fact on {subject} that looked like it corrected an older "
            "one, but that older fact had already been corrected by something else, so it "
            "left both alone."
        )
    why_it_matters = (
        "Both facts are currently shown as live on the note. If one is genuinely out of "
        "date, open the note and correct it by hand."
    )
    technical = _technical(
        row,
        extra={
            "note_slug": note_slug,
            "old_claim_id": payload.get("old_claim_id"),
            "new_claim_id": payload.get("new_claim_id") or payload.get("attempted_new_claim_id"),
        },
    )
    return ReviewCard(
        row_id=row["id"],
        kind="other",
        category_label=_CATEGORY_CONTRADICTION,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_basic_choices(),
        technical=technical,
        note_slug=note_slug,
        note_title=note_title,
        open_url=f"/notes/{note_slug}" if note_slug else None,
        open_label="Open the note →" if note_slug else None,
    )


def _present_no_home(row: dict[str, Any], payload: dict[str, Any]) -> ReviewCard:
    """A fact that names only the vendor's own side, from a source with no
    account to file it under (`merge.facts._queue_no_home`)."""
    mentions = ", ".join(f"“{m}”" for m in payload.get("mentions") or []) or "our own side"
    locator = payload.get("locator")
    source_label = _source_label(row.get("source"))
    open_url, open_label = _open_original(row, locator)
    fact_type = payload.get("fact_type")
    where = f"While reading {source_label}, " if source_label else ""
    headline = (
        f"{where}Portolan recorded a fact that names only our own side ({mentions}) and "
        "couldn't tell which customer it is about, so it isn't filed under any note yet."
    )
    why_it_matters = (
        "Nothing is lost -- the fact is kept here. If it matters, open the original and add "
        "it to the right account's note by hand."
    )
    return ReviewCard(
        row_id=row["id"],
        kind="other",
        category_label=_CATEGORY_OTHER,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_basic_choices(),
        technical=_technical(row, extra={"locator": locator}),
        claim_text=payload.get("claim"),
        quote=payload.get("quote"),
        fact_type_label=_FACT_TYPE_LABELS.get(fact_type) if fact_type else None,
        source_label=source_label,
        open_url=open_url,
        open_label=open_label,
    )


def _present_other_fallback(row: dict[str, Any], payload: dict[str, Any]) -> ReviewCard:
    """A row this module doesn't recognise: a plain-language sentence on the
    card, and the raw payload, one row per key as saved, in the technical
    fold."""
    headline = "Portolan flagged this for review but didn't record a plain-language reason."
    why_it_matters = (
        "Nothing is lost -- everything Portolan saved about it is in the technical details below."
    )
    technical = _technical(row)
    for key, value in payload.items():
        technical.append((key, _stringify(value)))
    return ReviewCard(
        row_id=row["id"],
        kind="other",
        category_label=_CATEGORY_OTHER,
        created=row.get("created"),
        headline=headline,
        why_it_matters=why_it_matters,
        choices=_basic_choices(),
        technical=technical,
    )


_OTHER_REASON_PRESENTERS: dict[str, Any] = {
    "numeral_not_supplied": _present_numeral_rejection,
    "resolved_language_without_a_resolved_theme": _present_resolved_language_rejection,
    "failed account synthesis": _present_failed_synthesis,
    "arr_not_parseable": _present_arr_not_parseable,
    "no_home_note": _present_no_home,
}


def _present_other(row: dict[str, Any]) -> ReviewCard:
    payload = row.get("payload") or {}
    reason = payload.get("reason")
    if reason == "contradicts verdict blocked: claim was already superseded":
        return _present_contradiction(row, payload, downgraded=False)
    if reason == _DOWNGRADED_CONTRADICTION_REASON:
        return _present_contradiction(row, payload, downgraded=True)
    presenter = _OTHER_REASON_PRESENTERS.get(reason)
    if presenter is not None:
        return presenter(row, payload)
    return _present_other_fallback(row, payload)


_KIND_PRESENTERS = {
    "unknown_entity": _present_unknown_entity,
    "failed_consolidation": _present_failed_consolidation,
    "theme_resolution": _present_theme_resolution,
    "quote_unsupported": _present_quote_unsupported,
    "owner_proposal": _present_owner_proposal,
    "action_card": _present_action_card,
    "action_done": _present_action_done,
    "other": _present_other,
}


def present_row(
    row: dict[str, Any], *, entities_by_crm_id: dict[str, dict[str, Any]] | None = None
) -> ReviewCard:
    """One enriched `review_queue` row -> one `ReviewCard`. `row` is a
    `review_queue_rows`-shaped dict plus any enrichment the route added:
    `source` (`sources_brief`), `context_snippet`, `note_title`,
    `evidence_backfill`, `proposal` (theme_resolution), `action_context`
    (action_card) and `done_context` (action_done). All optional: missing
    enrichment degrades the card, never raises. `entities_by_crm_id` names
    an `ambiguous_entity` row's candidates."""
    if row["kind"] == "ambiguous_entity":
        return _present_ambiguous_entity(row, entities_by_crm_id or {})
    presenter = _KIND_PRESENTERS.get(row["kind"])
    if presenter is not None:
        return presenter(row)
    # A kind added to the constraint before this module knows it.
    return _present_other_fallback(row, row.get("payload") or {})


def present_rows(
    rows: list[dict[str, Any]], *, entities_by_crm_id: dict[str, dict[str, Any]] | None = None
) -> list[ReviewCard]:
    return [present_row(row, entities_by_crm_id=entities_by_crm_id) for row in rows]


def category_label_for(kind: str, reason: str | None) -> str:
    """The `category_label` for a row of this `kind`/`reason`, from the
    index columns alone, so `/review` can count and filter the whole queue
    without presenting every row. It must agree with the label each
    presenter sets; the two are kept in step by hand.
    """
    if kind in _CATEGORY_BY_KIND:
        return _CATEGORY_BY_KIND[kind]
    if kind == "other":
        if reason in _CONTRADICTION_REASONS:
            return _CATEGORY_CONTRADICTION
        return _CATEGORY_BY_OTHER_REASON.get(reason, _CATEGORY_OTHER)
    # A kind this module doesn't know yet, as in `present_row`.
    return _CATEGORY_OTHER


def kind_counts(cards: list[ReviewCard]) -> list[tuple[str, int]]:
    """`[(category_label, count), ...]` over `cards`, in first-seen order."""
    counter: Counter[str] = Counter()
    order: list[str] = []
    for card in cards:
        if card.category_label not in counter:
            order.append(card.category_label)
        counter[card.category_label] += 1
    return [(label, counter[label]) for label in order]
