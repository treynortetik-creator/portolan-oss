"""Entity resolution and home-note selection for `merge_fact`
(`01-spec/build-contracts.md` "Merge").

Two things live here, both upstream of writing anything:

1. `resolve_fact_entities` -- turn `fact.entities` (mention strings or
   already-resolved registry ids) into `ResolvedMention`s, queuing unknowns
   and ambiguous-without-a-shared-parent mentions to `review_queue`.
2. `select_home` -- the Merge section's "pick the home note" table, plus a
   fallback for the gap it leaves: an account-anchored fact with no account
   entity (e.g. a competitor's own recall, no customer named) homes on the
   resolved product/competitor, `HomeChoice("entity", ...)`. It names the
   entity only; `merge_fact` decides which note that becomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from portolan.extract.schema import EntityKind, Fact, FactType
from portolan.ingest.models import SourceRow
from portolan.registry import Registry
from portolan.registry import queue_unknown as registry_queue_unknown
from portolan.registry.models import Resolution
from portolan.registry.registry import park_mention

# Fact types homed on the account's signal note.
ACCOUNT_ANCHORED_TYPES: frozenset[FactType] = frozenset(
    {
        FactType.COMPLAINT,
        FactType.REQUEST,
        FactType.PRAISE,
        FactType.RISK,
        FactType.COMPETITOR_MENTION,
        FactType.COMMITMENT,
        FactType.DECISION,
        FactType.PRICING,
        FactType.USAGE,
        FactType.QUESTION,
    }
)


@dataclass
class ResolvedMention:
    mention: str
    entity_id: str
    kind: EntityKind
    ambiguous: bool = False


@dataclass
class HomeChoice:
    kind: str  # "signal" | "contact" | "account" | "entity"
    entity_id: str


def resolve_mention_or_id(registry: Registry, mention: str, *, context_ids=(), as_of=None):
    """`registry.resolve()`, except that a string that is already a registry
    id resolves to itself: `resolve()` only matches names and aliases, so an
    id would otherwise be queued as unknown.
    """
    entity = registry.get(mention)
    if entity is not None:
        return Resolution(status="resolved", entity_id=entity.id, candidates=[entity.id])
    return registry.resolve(mention, context_ids=context_ids, as_of=as_of)


def _context_ids(source: SourceRow, registry: Registry) -> list[str]:
    """The source's participant entity ids (Merge section's `context_ids`),
    by email and by email domain: a raw `Participant` carries no
    `registry_id`."""
    ids: list[str] = []
    for participant in source.participants:
        if not participant.email:
            continue
        entity = registry.by_email(participant.email)
        if entity is not None:
            ids.append(entity.id)
        domain = participant.email.rsplit("@", 1)[-1] if "@" in participant.email else None
        if domain:
            ids.extend(e.id for e in registry.by_domain(domain))
    return list(dict.fromkeys(ids))


def _queue_ambiguous(
    conn: psycopg.Connection,
    mention: str,
    source_slug: str,
    candidates: list[str],
    fact: Fact,
    *,
    commit: bool = True,
) -> None:
    """Park an ambiguous mention with no shared parent (spec sec 3: "or
    parks in the review queue") as kind `ambiguous_entity`.

    The fact's claim, quote, locator and type are saved on the row so a
    reviewer can act on it. One open row per mention (`park_mention`): a
    later source is added to its `occurrences`; a repeat of the same source
    is a no-op.
    """
    occurrence = {
        "fact_type": fact.type.value,
        "claim": fact.claim,
        "quote": fact.quote,
        "locator": fact.locator,
        "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
    }
    park_mention(
        conn,
        kind="ambiguous_entity",
        mention=mention,
        source_slug=source_slug,
        payload={"mention": mention, "candidates": candidates, **occurrence},
        occurrence=occurrence,
    )
    if commit:
        conn.commit()


def resolve_fact_entities(
    conn: psycopg.Connection,
    registry: Registry,
    fact: Fact,
    source: SourceRow,
    *,
    ambiguous_mentions: list[str],
    commit: bool = True,
) -> tuple[list[ResolvedMention], int]:
    """Resolve every mention in `fact.entities`. Unknowns are queued
    (`registry.queue_unknown`); ambiguous mentions with a shared parent are
    attached to that parent and their mention text appended to
    `ambiguous_mentions` (caller marks the claim text, per the Merge
    section); ambiguous mentions with no shared parent are queued
    separately and dropped from the result -- facts are never dropped
    (spec sec 3), only the one mention's contribution to home-note
    selection and linking.
    """
    context_ids = _context_ids(source, registry)
    resolved: list[ResolvedMention] = []
    queued = 0
    # When the fact was said: its own world date when that is earlier than
    # the source (a CRM export recording a July departure), else the source.
    dates = [d for d in (fact.valid_from, source.source_date) if d is not None]
    said_on = min(dates) if dates else None

    for mention in fact.entities:
        result = resolve_mention_or_id(registry, mention, context_ids=context_ids, as_of=said_on)
        if result.status == "resolved" and result.entity_id:
            entity = registry.get(result.entity_id)
            if entity is not None:
                resolved.append(ResolvedMention(mention, entity.id, entity.kind))
        elif result.status == "ambiguous" and result.attach_to:
            entity = registry.get(result.attach_to)
            if entity is not None:
                resolved.append(ResolvedMention(mention, entity.id, entity.kind, ambiguous=True))
                ambiguous_mentions.append(mention)
        elif result.status == "ambiguous":
            _queue_ambiguous(conn, mention, source.slug, result.candidates, fact, commit=commit)
            queued += 1
        else:  # unknown
            # The fact's own text goes on the row; `fact.id` alone cannot be
            # looked up later.
            registry_queue_unknown(
                conn,
                mention,
                source.slug,
                {
                    "fact_id": fact.id,
                    "fact_type": fact.type.value,
                    "claim": fact.claim,
                    "quote": fact.quote,
                    "locator": fact.locator,
                    "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
                },
                commit=commit,
            )
            queued += 1

    return resolved, queued


def select_home(fact: Fact, resolved: list[ResolvedMention]) -> HomeChoice | None:
    """The Merge section's home-note table, plus this build's documented
    fallback (module docstring). `None` means nothing resolved that could
    anchor a home note; `merge_fact` then falls back to the source's
    account or parks the fact (its `fallback_account_id` docstring)."""
    account = next((r for r in resolved if r.kind == EntityKind.ACCOUNT), None)
    contact = next((r for r in resolved if r.kind == EntityKind.CONTACT), None)
    entity = next(
        (r for r in resolved if r.kind in (EntityKind.PRODUCT, EntityKind.COMPETITOR)), None
    )

    if fact.type == FactType.CONTACT_CHANGE:
        if contact:
            return HomeChoice("contact", contact.entity_id)
        if account:
            return HomeChoice("account", account.entity_id)
        if entity:
            return HomeChoice("entity", entity.entity_id)
        return None

    if fact.type in ACCOUNT_ANCHORED_TYPES:
        if account:
            return HomeChoice("signal", account.entity_id)
        if entity:  # no account entity at all (module docstring)
            return HomeChoice("entity", entity.entity_id)
        if contact:
            return HomeChoice("contact", contact.entity_id)
        return None

    # fact.type == FactType.OTHER
    if entity:
        return HomeChoice("entity", entity.entity_id)
    if account:
        return HomeChoice("account", account.entity_id)
    if contact:
        return HomeChoice("contact", contact.entity_id)
    return None
