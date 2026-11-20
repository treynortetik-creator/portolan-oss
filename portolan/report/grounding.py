"""The grounding gate for action cards (`01-spec/actions-and-owners-
2026-09-19.md`, "Phase 2, revised contract", "The gate").

`mechanical_checks` is byte and referential checks only: every evidence
item matches a live stored claim byte for byte and is in scope; every
finding was produced this run for this candidate; every reference index
is in range; every numeral in `what`/an inference sentence appears in
that sentence's referenced items; the owner is vendor-side and, when not
the default owner, named by the card's evidence or findings; the due
rule is one of the three the spec allows. The researcher's `submit_card`
runs the same function, so a model sees these errors and can retry.

`run_grounding_gate` runs `mechanical_checks`, then the typed judge
(`portolan.report.support_judge`) on `what` against all of the card's
references and on each inference sentence against its own references
only. Any failure or judge error drops the card and logs a `changelog`
row (kind `action_card_dropped`) with the reason codes; nothing rewrites
a card to make it pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import psycopg

from portolan.notes.persist import append_changelog
from portolan.notes.render import plain_text
from portolan.registry.owners import is_vendor_side_crm_id
from portolan.report.accounts import home_account_slug
from portolan.report.numbers import NumberFacts, check_numerals
from portolan.report.researcher import ActionCard, InferenceSentence
from portolan.report.support_judge import SUPPORT_JUDGE_MODEL, SUPPORT_JUDGE_SYSTEM, SupportJudgeFn

__all__ = ["GroundingResult", "due_errors", "mechanical_checks", "run_grounding_gate"]

CHANGELOG_KIND_DROPPED = "action_card_dropped"
NEXT_REPORT_DUE_WINDOW_DAYS = 21
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass
class GroundingResult:
    passed: bool
    reason: str | None = None
    checked_claim_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------
# Claim + scope helpers.
# ---------------------------------------------------------------------


def _load_claim(conn: psycopg.Connection, claim_id: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, note_slug, fact_type, claim_text, quote, source_slug, locator "
            "FROM claims WHERE id = %s AND archived = false AND superseded_at IS NULL",
            (claim_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    cid, note_slug, fact_type, claim_text, quote, source_slug, locator = row
    return {
        "id": cid,
        "note_slug": note_slug,
        "fact_type": fact_type,
        "claim_text": claim_text,
        "quote": quote,
        "source_slug": source_slug,
        "locator": locator,
    }


def _account_for_non_signal_note(conn: psycopg.Connection, note_slug: str) -> str | None:
    """A claim on a contact note resolves to the contact's parent
    account; one on an unregistered note (an action) resolves via an
    account or contact note that body-links to it."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT kind, parent_id FROM registry_entities WHERE note_slug = %s", (note_slug,)
        )
        row = cur.fetchone()
    if row is not None:
        kind, parent_id = row
        if kind == "contact" and parent_id is not None:
            with conn.cursor() as cur:
                cur.execute("SELECT note_slug FROM registry_entities WHERE id = %s", (parent_id,))
                parent_row = cur.fetchone()
            return parent_row[0] if parent_row else None
        return None

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.type, re.note_slug, parent_re.note_slug
            FROM links l
            JOIN notes n ON n.slug = l.from_slug
            LEFT JOIN registry_entities re ON re.note_slug = n.slug
            LEFT JOIN registry_entities parent_re ON parent_re.id = re.parent_id
            WHERE l.to_slug = %s AND l.kind = 'body'
            LIMIT 1
            """,
            (note_slug,),
        )
        link_row = cur.fetchone()
    if link_row is None:
        return None
    note_type, own_slug, parent_slug = link_row
    if note_type == "account":
        return own_slug
    if note_type == "contact":
        return parent_slug
    return None


def _resolve_claim_account(conn: psycopg.Connection, claim: dict[str, Any]) -> str | None:
    account = home_account_slug(
        conn,
        note_slug=claim["note_slug"],
        fact_type=claim["fact_type"],
        claim_text=claim["claim_text"],
    )
    if account is not None:
        return account
    return _account_for_non_signal_note(conn, claim["note_slug"])


def _claim_in_scope(
    conn: psycopg.Connection, claim: dict[str, Any], *, account_slug: str, theme_slug: str | None
) -> bool:
    if _resolve_claim_account(conn, claim) == account_slug:
        return True
    if theme_slug is None:
        return False
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM theme_members WHERE theme_slug = %s AND claim_id = %s",
            (theme_slug, claim["id"]),
        )
        return cur.fetchone() is not None


# ---------------------------------------------------------------------
# Mechanical checks (shared with `submit_card`).
# ---------------------------------------------------------------------


def _finding_exists(
    conn: psycopg.Connection, finding_id: int, *, run_id: str, candidate_slug: str
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM researcher_findings "
            "WHERE id = %s AND run_id = %s AND candidate_slug = %s",
            (finding_id, run_id, candidate_slug),
        )
        return cur.fetchone() is not None


def _load_finding_result(conn: psycopg.Connection, finding_id: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute("SELECT result FROM researcher_findings WHERE id = %s", (finding_id,))
        row = cur.fetchone()
    return row[0] if row else None


def _text_mentions(haystacks: list[str], needle: str) -> bool:
    needle_l = needle.strip().lower()
    if not needle_l:
        return False
    return any(needle_l in h.lower() for h in haystacks if h)


def _reference_text_parts(
    conn: psycopg.Connection, card: ActionCard, evidence_refs: list[int], finding_refs: list[int]
) -> list[str]:
    """The text a sentence's references license: each referenced evidence
    item's `text` and `quote`, and each referenced finding's loaded result
    (never its bare id, which would license the id as a number). Shared by
    the numeral gate and the judge phase so both see the same text."""
    parts: list[str] = []
    for i in evidence_refs:
        if 0 <= i < len(card.evidence):
            item = card.evidence[i]
            parts.append(plain_text(item.text))
            if item.quote:
                parts.append(item.quote)
    for j in finding_refs:
        if 0 <= j < len(card.findings):
            finding_id = card.findings[j]
            result = _load_finding_result(conn, finding_id) or {}
            parts.append(json.dumps(result, default=str))
    return parts


def _numeral_source_text(
    conn: psycopg.Connection, card: ActionCard, evidence_refs: list[int], finding_refs: list[int]
) -> str:
    return " ".join(_reference_text_parts(conn, card, evidence_refs, finding_refs))


def due_errors(card: ActionCard, *, as_of: date) -> list[str]:
    """The due-date rules on their own: required for an `action`
    verdict; when present, its rule must be one of the three the spec
    allows, with that rule's window measured from `as_of`. Separate from
    `mechanical_checks` so Confirm can tell a stale due date apart from
    changed evidence."""
    errors: list[str] = []
    if card.verdict == "action" and card.due is None:
        errors.append("due_missing")
    if card.due is not None:
        if not card.due_rule:
            errors.append("due_rule_missing")
        else:
            rule = card.due_rule.strip()
            evidence_haystacks = [item.text for item in card.evidence] + [
                item.quote or "" for item in card.evidence
            ]
            if rule == "next_report":
                if not (as_of < card.due <= as_of + timedelta(days=NEXT_REPORT_DUE_WINDOW_DAYS)):
                    errors.append("due_outside_next_report_window")
            elif rule.startswith("renewal_minus_days:"):
                # `due + N` must be a renewal date named in the evidence;
                # 0 <= N <= 365 keeps `due` on or before that renewal, and
                # `due` may sit at most one `next_report` window in the past.
                try:
                    n_days = int(rule.split(":", 1)[1])
                except ValueError:
                    errors.append(f"due_rule_invalid_format:{rule}")
                else:
                    if not (0 <= n_days <= 365):
                        errors.append(f"renewal_minus_days_out_of_range:{n_days}")
                    renewal_date = card.due + timedelta(days=n_days)
                    if not _text_mentions(evidence_haystacks, renewal_date.isoformat()):
                        errors.append("renewal_date_not_in_evidence")
                    window_start = as_of - timedelta(days=NEXT_REPORT_DUE_WINDOW_DAYS)
                    if not (window_start <= card.due <= renewal_date):
                        errors.append("due_outside_renewal_window")
            elif rule == "commitment_date":
                # A past commitment date is allowed (an overdue item); a
                # future one may be at most one `next_report` window ahead.
                if not _text_mentions(evidence_haystacks, card.due.isoformat()):
                    errors.append("commitment_date_not_in_evidence")
                if card.due > as_of + timedelta(days=NEXT_REPORT_DUE_WINDOW_DAYS):
                    errors.append("commitment_date_too_far_in_future")
            else:
                errors.append(f"due_rule_unrecognized:{rule}")

    return errors


def mechanical_checks(
    conn: psycopg.Connection,
    card: ActionCard,
    *,
    default_owner_crm_id: str | None,
    as_of: date,
    check_due: bool = True,
) -> list[str]:
    """Every check except the typed judge. Returns every error found,
    empty when the card is clean; an unknown account returns that one
    error alone. `check_due=False` skips only `due_errors` (a reviewer's
    own due date replaces the researcher's proposal)."""
    errors: list[str] = []

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM notes WHERE slug = %s AND type = 'account'", (card.account,))
        if cur.fetchone() is None:
            return [f"account_not_found:{card.account}"]
    if card.theme is not None:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM notes WHERE slug = %s AND type = 'theme'", (card.theme,))
            if cur.fetchone() is None:
                errors.append(f"theme_not_found:{card.theme}")

    # Any verdict: a near-blank `what` never reaches the judge or Confirm.
    if len(card.what.strip()) < 20:
        errors.append("what_too_short")

    if not card.evidence and not card.findings:
        errors.append("no_evidence_or_findings")
    # An `action` must cite the claim it acts on; `no_action`/`trigger_stale`
    # may rest on findings alone (often a computed absence).
    if card.verdict == "action" and not card.evidence:
        errors.append("action_needs_evidence")

    # Evidence: byte-equal to a live stored claim and in scope (the card's
    # account, or its theme). An `action` also needs one item from its own
    # account: a theme-scoped claim can belong to another customer.
    has_own_account_evidence = False
    for item in card.evidence:
        stored = _load_claim(conn, item.claim_id)
        if stored is None:
            errors.append(f"claim_not_found:{item.claim_id}")
            continue
        if (
            stored["claim_text"] != item.text
            or (stored["quote"] or None) != (item.quote or None)
            or stored["source_slug"] != item.source_slug
            or stored["locator"] != item.locator
        ):
            errors.append(f"evidence_not_verbatim:{item.claim_id}")
            continue
        if not _claim_in_scope(conn, stored, account_slug=card.account, theme_slug=card.theme):
            errors.append(f"claim_out_of_scope:{item.claim_id}")
            continue
        if _resolve_claim_account(conn, stored) == card.account:
            has_own_account_evidence = True
    if card.verdict == "action" and card.evidence and not has_own_account_evidence:
        errors.append("no_own_account_evidence")

    # Findings: must exist for THIS run and candidate -- the model cannot
    # invent a finding_id it was never handed.
    for finding_id in card.findings:
        if not _finding_exists(
            conn, finding_id, run_id=card.run_id, candidate_slug=card.candidate_slug
        ):
            errors.append(f"finding_not_found:{finding_id}")

    n_evidence = len(card.evidence)
    n_findings = len(card.findings)

    def _refs_in_range(refs: InferenceSentence | Any, label: str) -> None:
        for i in refs.evidence_refs:
            if not (0 <= i < n_evidence):
                errors.append(f"ref_out_of_range:{label}:evidence:{i}")
        for j in refs.finding_refs:
            if not (0 <= j < n_findings):
                errors.append(f"ref_out_of_range:{label}:finding:{j}")

    for idx, sentence in enumerate(card.inference):
        _refs_in_range(sentence, f"inference:{idx}")
    _refs_in_range(card.confidence_refs, "confidence")

    # Owner: vendor-side, and justified when it differs from the
    # candidate's own default account owner.
    if card.verdict == "action" and card.owner is None:
        errors.append("owner_missing")
    if card.owner is not None:
        if not is_vendor_side_crm_id(conn, card.owner):
            errors.append(f"owner_not_vendor_side:{card.owner}")
        elif card.owner != default_owner_crm_id:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT canonical_name FROM registry_entities WHERE crm_id = %s", (card.owner,)
                )
                name_row = cur.fetchone()
            owner_name = name_row[0] if name_row else None
            haystacks = [item.text for item in card.evidence] + [
                item.quote or "" for item in card.evidence
            ]
            for finding_id in card.findings:
                result = _load_finding_result(conn, finding_id)
                if result is not None:
                    haystacks.append(json.dumps(result))
            owner_mentioned = _text_mentions(haystacks, card.owner) or (
                owner_name and _text_mentions(haystacks, owner_name)
            )
            if not owner_mentioned:
                errors.append("owner_not_justified")

    if check_due:
        errors.extend(due_errors(card, as_of=as_of))

    # Numerals: `what` against ALL evidence+findings; each inference
    # sentence against only its own referenced subset.
    all_evidence_refs = list(range(n_evidence))
    all_finding_refs = list(range(n_findings))
    what_facts = NumberFacts()
    what_facts.harvest(_numeral_source_text(conn, card, all_evidence_refs, all_finding_refs))
    what_violations = check_numerals(card.what, what_facts, present_markers=set())
    if what_violations:
        errors.append(f"numeral_not_supported:what:{','.join(what_violations)}")

    for idx, sentence in enumerate(card.inference):
        facts = NumberFacts()
        facts.harvest(
            _numeral_source_text(conn, card, sentence.evidence_refs, sentence.finding_refs)
        )
        violations = check_numerals(sentence.sentence, facts, present_markers=set())
        if violations:
            errors.append(f"numeral_not_supported:inference:{idx}:{','.join(violations)}")

    return errors


# ---------------------------------------------------------------------
# The typed judge phase, with its cache.
# ---------------------------------------------------------------------


def _sentence_hash(sentence: str) -> str:
    return hashlib.sha256(sentence.strip().encode()).hexdigest()


def _reference_text_and_hash(
    conn: psycopg.Connection, card: ActionCard, evidence_refs: list[int], finding_refs: list[int]
) -> tuple[str, str]:
    reference_text = " ".join(_reference_text_parts(conn, card, evidence_refs, finding_refs))
    hash_parts: list[str] = []
    for i in evidence_refs:
        if 0 <= i < len(card.evidence):
            item = card.evidence[i]
            hash_parts.append(f"e:{item.claim_id}:{item.text}:{item.quote or ''}")
    for j in finding_refs:
        if 0 <= j < len(card.findings):
            finding_id = card.findings[j]
            result = _load_finding_result(conn, finding_id) or {}
            hash_parts.append(f"f:{finding_id}:{json.dumps(result, default=str, sort_keys=True)}")
    reference_hash = hashlib.sha256("|".join(hash_parts).encode()).hexdigest()
    return reference_text, reference_hash


def _judge_version(model: str) -> str:
    return hashlib.sha256(f"{SUPPORT_JUDGE_SYSTEM}|{model}".encode()).hexdigest()


def _cached_verdict(
    conn: psycopg.Connection, *, sentence_hash: str, reference_hash: str, judge_version: str
) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT verdict FROM support_judge_verdicts "
            "WHERE sentence_hash = %s AND reference_hash = %s AND judge_version = %s",
            (sentence_hash, reference_hash, judge_version),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _store_verdict(
    conn: psycopg.Connection,
    *,
    sentence: str,
    sentence_hash: str,
    reference_hash: str,
    judge_version: str,
    verdict: str,
    commit: bool,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO support_judge_verdicts
                (sentence_hash, reference_hash, judge_version, sentence, verdict)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (sentence_hash, reference_hash, judge_version)
            DO UPDATE SET verdict = EXCLUDED.verdict, judged_at = now()
            """,
            (sentence_hash, reference_hash, judge_version, sentence, verdict),
        )
    if commit:
        conn.commit()


def _judged_support(
    conn: psycopg.Connection,
    judge: SupportJudgeFn,
    *,
    sentence: str,
    reference_text: str,
    reference_hash: str,
    judge_version: str,
    commit: bool,
) -> str:
    """`"supported"` / `"not_supported"` / `"error"`."""
    sentence_hash = _sentence_hash(sentence)
    cached = _cached_verdict(
        conn,
        sentence_hash=sentence_hash,
        reference_hash=reference_hash,
        judge_version=judge_version,
    )
    if cached is not None:
        return cached
    verdict = judge(sentence=sentence, reference_text=reference_text)
    if verdict is None:
        return "error"
    _store_verdict(
        conn,
        sentence=sentence,
        sentence_hash=sentence_hash,
        reference_hash=reference_hash,
        judge_version=judge_version,
        verdict=verdict,
        commit=commit,
    )
    return verdict


def _log_dropped(conn: psycopg.Connection, card: ActionCard, reason: str, *, commit: bool) -> None:
    append_changelog(
        conn,
        CHANGELOG_KIND_DROPPED,
        f"action card dropped by the grounding gate: {reason}",
        {
            "candidate_slug": card.candidate_slug,
            "account": card.account,
            "theme": card.theme,
            "verdict": card.verdict,
            "reason": reason,
            "what": card.what,
        },
        commit=commit,
    )


def run_grounding_gate(
    conn: psycopg.Connection,
    card: ActionCard,
    *,
    judge: SupportJudgeFn,
    as_of: date,
    default_owner_crm_id: str | None = None,
    judge_model: str = SUPPORT_JUDGE_MODEL,
    commit: bool = True,
) -> GroundingResult:
    errors = mechanical_checks(conn, card, default_owner_crm_id=default_owner_crm_id, as_of=as_of)
    if errors:
        reason = "; ".join(errors)
        _log_dropped(conn, card, reason, commit=commit)
        return GroundingResult(passed=False, reason=reason)

    judge_version = _judge_version(judge_model)
    checked_claim_ids = [item.claim_id for item in card.evidence]

    all_evidence_refs = list(range(len(card.evidence)))
    all_finding_refs = list(range(len(card.findings)))
    what_text, what_hash = _reference_text_and_hash(conn, card, all_evidence_refs, all_finding_refs)
    what_verdict = _judged_support(
        conn,
        judge,
        sentence=card.what,
        reference_text=what_text,
        reference_hash=what_hash,
        judge_version=judge_version,
        commit=commit,
    )
    if what_verdict == "error":
        result = GroundingResult(
            passed=False, reason="judge_error:what", checked_claim_ids=checked_claim_ids
        )
        _log_dropped(conn, card, result.reason, commit=commit)
        return result
    if what_verdict != "supported":
        result = GroundingResult(
            passed=False, reason="judge_not_supported:what", checked_claim_ids=checked_claim_ids
        )
        _log_dropped(conn, card, result.reason, commit=commit)
        return result

    for idx, sentence in enumerate(card.inference):
        ref_text, ref_hash = _reference_text_and_hash(
            conn, card, sentence.evidence_refs, sentence.finding_refs
        )
        verdict = _judged_support(
            conn,
            judge,
            sentence=sentence.sentence,
            reference_text=ref_text,
            reference_hash=ref_hash,
            judge_version=judge_version,
            commit=commit,
        )
        if verdict == "error":
            result = GroundingResult(
                passed=False,
                reason=f"judge_error:inference:{idx}",
                checked_claim_ids=checked_claim_ids,
            )
            _log_dropped(conn, card, result.reason, commit=commit)
            return result
        if verdict != "supported":
            result = GroundingResult(
                passed=False,
                reason=f"judge_not_supported:inference:{idx}",
                checked_claim_ids=checked_claim_ids,
            )
            _log_dropped(conn, card, result.reason, commit=commit)
            return result

    return GroundingResult(passed=True, checked_claim_ids=checked_claim_ids)
