"""`customer_filter`: is this source worth extracting? (spec sec 5.2)

"The crawl reads customer-touching sources, not the whole company."
Rules run in order and the first one that fires decides the source:

1. junk (auto-replies, out-of-office, calendar invites, newsletters,
   `noreply@`) -> excluded, `junk:<kind>`; enters the same seeded sample as
   any other exclusion, so a junk misclassification can still be audited.
2. a participant's email or email domain resolves to a registry
   `account` or `contact` -> included, `participant_domain`.
3. `raw.account_hints` is non-empty (the connector already flagged an
   account) -> included, `account_hint`.
4. `registry.scan(prepared.text)` finds an account/contact/product/
   competitor mention -> included, `registry_scan`.
5. a signal word appears (`SIGNAL_WORDS`) *and* the source isn't a
   conversation entirely among our own people -> included, `signal_word`.
6. otherwise excluded (`excluded`); a deterministic seeded sample is
   flagged `excluded_sampled` so the miss-rate audit (`portolan-spec-v2.md`
   sec 6.4) has something to re-extract.

Rule 5's carve-out: an internal email can use a signal word by coincidence
("delivery" in a note about a branch driver shortage). The truth table
wants that source excluded and caught by the sample audit
(`corpus/truth-table.yaml`, `sample_audit_should_find: [F07]`). One
participant the registry cannot place is enough to count the signal word.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from portolan.extract.schema import EntityKind
from portolan.ingest.models import Participant, PreparedText, RawSource

__all__ = [
    "SIGNAL_WORDS",
    "FilterDecision",
    "Registry",
    "RegistryEntityLike",
    "RegistryMatchLike",
    "apply_filter_decision",
    "customer_filter",
]


# --------------------------------------------------------------------
# The slice of the registry interface this module calls (`by_email`,
# `by_domain`, `scan`), as Protocols the real registry satisfies
# structurally.
# --------------------------------------------------------------------


class RegistryEntityLike(Protocol):
    id: str
    kind: EntityKind


class RegistryMatchLike(Protocol):
    entity_id: str
    kind: EntityKind


class Registry(Protocol):
    def by_email(self, email: str) -> RegistryEntityLike | None: ...

    def by_domain(self, domain: str) -> list[RegistryEntityLike]: ...

    def scan(self, text: str) -> list[RegistryMatchLike]: ...


class FilterDecision(BaseModel):
    """build-contracts.md: `include`, `reason`, `matched_entities`,
    `sampled`. `reason` is a short machine-readable code (module docstring)."""

    model_config = ConfigDict(extra="forbid")

    include: bool
    reason: str
    matched_entities: list[str] = Field(default_factory=list)
    sampled: bool = False


# --------------------------------------------------------------------
# 1. Junk rules
# --------------------------------------------------------------------

_AUTO_REPLY_SUBJECT_RE = re.compile(
    r"^(automatic reply|auto[- ]?reply|out of office|ooo)\b", re.IGNORECASE
)
_OUT_OF_OFFICE_BODY_RE = re.compile(r"\bout of (the )?office\b", re.IGNORECASE)
_CALENDAR_INVITE_SUBJECT_RE = re.compile(r"^(invitation|invite):", re.IGNORECASE)
_CALENDAR_CONTENT_TYPES = ("text/calendar",)
_NEWSLETTER_SUBJECT_RE = re.compile(r"\b(newsletter|weekly digest|unsubscribe)\b", re.IGNORECASE)
_NOREPLY_RE = re.compile(r"^no-?reply@", re.IGNORECASE)


def _out_of_office_in_body(text: str) -> bool:
    """Only the first two lines count: a real auto-reply opens with the
    phrase, while a customer may write "I'll be out of the office next week"
    mid-message."""
    first_two_lines = "\n".join(text.splitlines()[:2])
    return bool(_OUT_OF_OFFICE_BODY_RE.search(first_two_lines))


def _junk_reason(raw: RawSource) -> str | None:
    subject = str(raw.metadata.get("subject", ""))
    content_type = str(raw.metadata.get("content_type", ""))
    from_email = str(raw.metadata.get("from_email", ""))

    if from_email and _NOREPLY_RE.match(from_email):
        return "junk:noreply"

    if _AUTO_REPLY_SUBJECT_RE.match(subject) or _out_of_office_in_body(raw.text):
        return "junk:auto_reply"

    if _CALENDAR_INVITE_SUBJECT_RE.match(subject) or content_type in _CALENDAR_CONTENT_TYPES:
        return "junk:calendar_invite"

    if _NEWSLETTER_SUBJECT_RE.search(subject):
        return "junk:newsletter"

    return None


# --------------------------------------------------------------------
# 2. Participant email/domain against the registry
# --------------------------------------------------------------------

_CUSTOMER_KINDS = {EntityKind.ACCOUNT, EntityKind.CONTACT}


def _domain_of(email: str) -> str:
    return email.rsplit("@", 1)[-1].lower()


def _participant_registry_match(
    participants: list[Participant], registry: Registry
) -> RegistryEntityLike | None:
    for participant in participants:
        if not participant.email:
            continue
        entity = registry.by_email(participant.email)
        if entity is not None and entity.kind in _CUSTOMER_KINDS:
            return entity
        for domain_entity in registry.by_domain(_domain_of(participant.email)):
            if domain_entity.kind in _CUSTOMER_KINDS:
                return domain_entity
    return None


def _all_participants_internal(participants: list[Participant], registry: Registry) -> bool:
    """True only when every participant with an email resolves, by exact
    email or by domain, to nothing but `internal_person` registry
    entities — i.e. this is a conversation entirely among our own people,
    with no customer, and no *unplaced* external address either. A
    participant the registry has never heard of (a personal address, an
    unregistered customer domain) is NOT "internal" — it just isn't a
    known customer yet, which is exactly the ambiguous case the
    registry-scan/signal-word rules exist to catch on the text itself.
    """
    if not participants:
        return False
    for participant in participants:
        if not participant.email:
            return False
        entity = registry.by_email(participant.email)
        if entity is not None:
            if entity.kind != EntityKind.INTERNAL_PERSON:
                return False
            continue
        domain_entities = registry.by_domain(_domain_of(participant.email))
        if not domain_entities or not all(
            e.kind == EntityKind.INTERNAL_PERSON for e in domain_entities
        ):
            return False
    return True


# --------------------------------------------------------------------
# 4/5. Registry scan and the signal-word list
# --------------------------------------------------------------------

SIGNAL_WORDS: tuple[str, ...] = (
    "delivery",
    "late",
    "escalat",
    "renew",
    "cancel",
    "churn",
    "complaint",
    "outage",
    "recall",
)


def _find_signal_word(text: str) -> str | None:
    low = text.lower()
    for word in SIGNAL_WORDS:
        if re.search(rf"\b{re.escape(word)}", low):
            return word
    return None


# --------------------------------------------------------------------
# 6. Deterministic seeded sampling of exclusions
# --------------------------------------------------------------------


def _seeded_sample(source_id: str, sample_rate: float, seed: int | str | None) -> bool:
    """Deterministic given `(seed, source_id)`, so a fixed seed reproduces
    exactly the same sampled set across runs — required for the coverage
    ledger's `sampled` count to be meaningful and for tests to assert a
    specific source lands in (or out of) the sample.
    """
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    return fraction < sample_rate


# --------------------------------------------------------------------
# The filter
# --------------------------------------------------------------------


def customer_filter(
    raw: RawSource,
    prepared: PreparedText,
    registry: Registry,
    *,
    sample_rate: float = 0.02,
    seed: int | str | None = None,
) -> FilterDecision:
    junk_reason = _junk_reason(raw)
    if junk_reason is not None:
        # Junk is sampled like any exclusion, so a misclassification is auditable.
        sampled = _seeded_sample(raw.source_id, sample_rate, seed)
        return FilterDecision(
            include=False, reason=junk_reason, matched_entities=[], sampled=sampled
        )

    participant_match = _participant_registry_match(raw.participants, registry)
    if participant_match is not None:
        return FilterDecision(
            include=True,
            reason="participant_domain",
            matched_entities=[participant_match.id],
            sampled=False,
        )

    if raw.account_hints:
        return FilterDecision(
            include=True,
            reason="account_hint",
            matched_entities=list(raw.account_hints),
            sampled=False,
        )

    scan_matches = registry.scan(prepared.text)
    # The vendor's own side (its people, itself, its branches: kind `other`)
    # says nothing about a customer.
    entity_matches = [
        m for m in scan_matches if m.kind not in (EntityKind.INTERNAL_PERSON, EntityKind.OTHER)
    ]
    if entity_matches:
        matched_ids = list(dict.fromkeys(m.entity_id for m in entity_matches))
        return FilterDecision(
            include=True, reason="registry_scan", matched_entities=matched_ids, sampled=False
        )

    if not _all_participants_internal(raw.participants, registry):
        signal_word = _find_signal_word(prepared.text)
        if signal_word is not None:
            return FilterDecision(
                include=True, reason="signal_word", matched_entities=[signal_word], sampled=False
            )

    sampled = _seeded_sample(raw.source_id, sample_rate, seed)
    reason = "excluded_sampled" if sampled else "excluded"
    return FilterDecision(include=False, reason=reason, matched_entities=[], sampled=sampled)


def apply_filter_decision(conn, slug: str, decision: FilterDecision) -> None:
    """Apply a `FilterDecision` to the `sources` row and commit: `filtered`
    with `filter_reason` when excluded (sampled or not; the ledger counts
    samples), `pending` with no reason when included.

    Does not write `coverage_ledger`; the caller records the decision there
    (`tests/test_ingest_filter.py::test_apply_filter_decision_does_not_touch_the_coverage_ledger`).
    """
    status = "pending" if decision.include else "filtered"
    filter_reason = None if decision.include else decision.reason
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE sources SET extraction_status = %s, filter_reason = %s WHERE slug = %s",
            (status, filter_reason, slug),
        )
    conn.commit()
