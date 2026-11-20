"""Owner assignment (`01-spec/actions-and-owners-2026-09-19.md`, Phase 1:
"Owner is a registry fact, not note text").

Two evidence sources, one ranking rule. The assignment rule, verbatim:
"rank the vendor-side contacts ... by the number of that account's
sources they appear in, most recent first, ties broken by commitments
they made for that account."

- `evidence_from_corpus` / `assign_owners_from_corpus`: the synthetic-
  world path (`python -m portolan.registry assign-owners --corpus
  corpus/out`). Reads `corpus/out/manifest.json` (source participants,
  accounts, dates -- ground truth, not extraction output) and
  `corpus/truth-table.yaml` (which facts are commitments, and by whom),
  via `portolan.bench.truth`, so it does not depend on extraction quality.

- `evidence_from_claims` / `propose_owners`: the real-customer path.
  Reads the live graph only -- no manifest, no truth table. (This module
  still imports `portolan.bench.truth` at load, for the corpus path's
  types.)

Both compute the same shape of evidence (`OwnerCandidate`: source count,
last seen, commitments) and rank it with the same `rank_candidates`.

**Vendor-side only**: an owner is always a registry entity of kind
`internal_person`. `is_vendor_side`/`is_vendor_side_crm_id` hold that
check; `confirm_owner_proposal` applies it. `backfill_action_owners`
copies an account's already-confirmed owner and does not re-check it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import psycopg

from portolan.bench.truth import Account, Manifest, TruthTable
from portolan.registry.registry import Registry

__all__ = [
    "AssignmentResult",
    "InternalPeopleIndex",
    "OwnerCandidate",
    "ProposalRow",
    "account_note_slugs",
    "assign_owners_from_corpus",
    "backfill_action_owners",
    "confirm_owner_proposal",
    "evidence_from_claims",
    "evidence_from_corpus",
    "is_vendor_side",
    "is_vendor_side_crm_id",
    "load_internal_people_index",
    "propose_owners",
    "rank_candidates",
    "write_owner_crm_id_column",
]


@dataclass
class OwnerCandidate:
    """One vendor-side person's evidence for owning one account."""

    person_crm_id: str
    name: str
    source_count: int
    last_seen: date | None
    commitments: int


def _sort_key(c: OwnerCandidate) -> tuple[int, int, int]:
    # Most sources first; ties broken by the most recent source; further
    # ties broken by the most commitments made for this account. `None`
    # (never seen) sorts after every real date.
    last_seen_ordinal = c.last_seen.toordinal() if c.last_seen is not None else -1
    return (-c.source_count, -last_seen_ordinal, -c.commitments)


def rank_candidates(candidates: list[OwnerCandidate]) -> list[OwnerCandidate]:
    """`candidates`, ranked highest-evidence first (module docstring's
    ranking rule). A tie in every field is broken by `person_crm_id` so
    the order is fully deterministic (never dependent on dict iteration
    order)."""
    return sorted(candidates, key=lambda c: (*_sort_key(c), c.person_crm_id))


@dataclass
class AssignmentResult:
    """One account's resolved owner (`assign_owners_from_corpus`):
    the evidence that won, plus `basis` -- "own evidence", "inherited
    from parent (<name>)", or "territory fallback (<region>)" -- so the
    printed table can show *why*, not just *who*."""

    account_id: str
    account_crm_id: str
    account_name: str
    owner_crm_id: str
    owner_name: str
    source_count: int
    last_seen: date | None
    commitments: int
    basis: str


# ---------------------------------------------------------------------
# The synthetic-world path: corpus/out/manifest.json + truth-table.yaml.
# ---------------------------------------------------------------------


def evidence_from_corpus(table: TruthTable, manifest: Manifest) -> dict[str, list[OwnerCandidate]]:
    """`{account_id: [OwnerCandidate, ...]}`, ranked, for every account
    the manifest's sources name with a vendor rep present. An account with
    no entry here has zero direct evidence (`assign_owners_from_corpus`
    applies inheritance/fallback for those).

    "Appears in a source" is manifest participation (`ManifestSource.
    participants`), not "spoke the one planted fact" -- a rep sitting in
    on a call without being quoted still counts. "Commitments they made
    for that account" is narrower: truth-table facts of type `commitment`
    whose `entities` include the account and whose `sources` name the
    person as `speaker`. A fact's `entities` can also name the person who
    committed (e.g. `[ashgrove, sam-okafor]`); `account_ids` filters those
    out so a person's id is never taken for an account.
    """
    internal_people = table.meta.get("internal_people") or []
    internal_ids = {p["id"] for p in internal_people}
    name_by_id = {p["id"]: p["name"] for p in internal_people}
    account_ids = {a.id for a in table.accounts}

    # account_id -> person_id -> {"sources": {source_id, ...}, "last_seen": date|None}
    seen: dict[str, dict[str, dict[str, Any]]] = {}
    for src in manifest.sources:
        present = [p for p in src.participants if p in internal_ids]
        if not present:
            continue
        for account_id in src.accounts:
            if account_id not in account_ids:
                continue
            bucket = seen.setdefault(account_id, {})
            for person_id in present:
                stats = bucket.setdefault(person_id, {"sources": set(), "last_seen": None})
                stats["sources"].add(src.id)
                if src.date is not None and (
                    stats["last_seen"] is None or src.date > stats["last_seen"]
                ):
                    stats["last_seen"] = src.date

    # account_id -> person_id -> commitment count
    commitments: dict[str, dict[str, int]] = {}
    for fact in table.facts:
        if fact.type != "commitment":
            continue
        speakers = {s.speaker for s in fact.sources if s.speaker}
        people_here = speakers & internal_ids
        if not people_here:
            continue
        for account_id in fact.entities:
            if account_id not in account_ids:
                continue
            bucket = commitments.setdefault(account_id, {})
            for person_id in people_here:
                bucket[person_id] = bucket.get(person_id, 0) + 1

    result: dict[str, list[OwnerCandidate]] = {}
    for account_id, bucket in seen.items():
        candidates = [
            OwnerCandidate(
                person_crm_id=person_id,
                name=name_by_id.get(person_id, person_id),
                source_count=len(stats["sources"]),
                last_seen=stats["last_seen"],
                commitments=commitments.get(account_id, {}).get(person_id, 0),
            )
            for person_id, stats in bucket.items()
        ]
        result[account_id] = rank_candidates(candidates)
    return result


def _resolve_one(
    account: Account,
    evidence: dict[str, list[OwnerCandidate]],
    *,
    territory_owner: dict[str, str],
    name_by_id: dict[str, str],
) -> AssignmentResult:
    candidates = evidence.get(account.id) or []
    if candidates:
        top = candidates[0]
        return AssignmentResult(
            account_id=account.id,
            account_crm_id=account.crm_id or account.id,
            account_name=account.name,
            owner_crm_id=top.person_crm_id,
            owner_name=top.name,
            source_count=top.source_count,
            last_seen=top.last_seen,
            commitments=top.commitments,
            basis="own evidence",
        )
    fallback_id = territory_owner.get(account.region or "")
    if fallback_id is None:
        raise ValueError(
            f"no owner evidence and no territory fallback for account {account.id!r} "
            f"(region={account.region!r})"
        )
    return AssignmentResult(
        account_id=account.id,
        account_crm_id=account.crm_id or account.id,
        account_name=account.name,
        owner_crm_id=fallback_id,
        owner_name=name_by_id.get(fallback_id, fallback_id),
        source_count=0,
        last_seen=None,
        commitments=0,
        basis=f"territory fallback ({account.region})",
    )


def assign_owners_from_corpus(table: TruthTable, manifest: Manifest) -> dict[str, AssignmentResult]:
    """Every account's assigned owner (the synthetic-world path), complete
    or `ValueError`: own evidence first, then (for a `facility` with a
    `parent`) the parent's resolved owner, then the internal person whose
    `territory` matches the account's `region`. A region with no matching
    territory raises rather than leaving an account silently unowned.

    "Child facilities inherit the parent system's owner unless their own
    evidence says otherwise": only `kind == "facility"` accounts with a
    `parent` inherit. A `system` under a `capital_partner` (e.g. Larkspur
    Regional Health under Pinecrest Health Partners) is resolved on its
    own, like a parent-less account.
    """
    evidence = evidence_from_corpus(table, manifest)
    internal_people = table.meta.get("internal_people") or []
    name_by_id = {p["id"]: p["name"] for p in internal_people}
    territory_owner: dict[str, str] = {
        p["territory"]: p["id"] for p in internal_people if p.get("territory")
    }

    results: dict[str, AssignmentResult] = {}
    accounts_by_id = {a.id: a for a in table.accounts}

    independent = [a for a in table.accounts if not (a.parent and a.kind == "facility")]
    dependent = [a for a in table.accounts if a.parent and a.kind == "facility"]

    for account in independent:
        results[account.id] = _resolve_one(
            account, evidence, territory_owner=territory_owner, name_by_id=name_by_id
        )

    for account in dependent:
        candidates = evidence.get(account.id) or []
        if candidates:
            results[account.id] = _resolve_one(
                account, evidence, territory_owner=territory_owner, name_by_id=name_by_id
            )
            continue
        parent = results.get(account.parent or "")
        if parent is None:
            # The parent is a facility this loop has not resolved yet: resolve it on its own
            # evidence or territory rather than recursing.
            parent_account = accounts_by_id[account.parent] if account.parent else account
            parent = _resolve_one(
                parent_account, evidence, territory_owner=territory_owner, name_by_id=name_by_id
            )
        results[account.id] = AssignmentResult(
            account_id=account.id,
            account_crm_id=account.crm_id or account.id,
            account_name=account.name,
            owner_crm_id=parent.owner_crm_id,
            owner_name=parent.owner_name,
            source_count=0,
            last_seen=None,
            commitments=0,
            basis=f"inherited from parent ({parent.account_name})",
        )

    missing = [a.id for a in table.accounts if a.id not in results]
    if missing:
        raise ValueError(f"no owner resolved for accounts: {missing}")
    return results


def write_owner_crm_id_column(accounts_csv: Path, assignments: dict[str, AssignmentResult]) -> None:
    """Add (or overwrite) an `owner_crm_id` column on `accounts_csv`, one
    per row, keyed by that row's own `crm_id` (idempotent: a second run
    just overwrites the column's values, never appends a duplicate
    column)."""
    with accounts_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    by_crm_id = {a.account_crm_id: a for a in assignments.values()}
    if "owner_crm_id" not in fieldnames:
        fieldnames = [*fieldnames, "owner_crm_id"]

    for row in rows:
        assignment = by_crm_id.get(row.get("crm_id", ""))
        row["owner_crm_id"] = assignment.owner_crm_id if assignment else ""

    with accounts_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# The real-customer path: the live graph (claims/sources/registry_entities).
# No `portolan.bench.truth` import below this line -- a real customer's
# graph has neither a manifest nor a truth table.
# ---------------------------------------------------------------------


def is_vendor_side(kind: str) -> bool:
    """An owner is always a vendor-side person: registry kind `internal_person`."""
    return kind == "internal_person"


def is_vendor_side_crm_id(conn: psycopg.Connection, crm_id: str) -> bool:
    """`is_vendor_side`, looked up by `registry_entities.crm_id` -- `False`
    for an id that doesn't resolve to any registry entity at all, same as
    for one that resolves but isn't `internal_person`."""
    with conn.cursor() as cur:
        cur.execute("SELECT kind FROM registry_entities WHERE crm_id = %s", (crm_id,))
        row = cur.fetchone()
    return row is not None and is_vendor_side(row[0])


@dataclass
class InternalPeopleIndex:
    """`by_key`: canonical name or lower-cased email -> `crm_id`, for
    recognising a vendor rep among a source's `participants`/`speaker`.
    `name_by_id`: `crm_id` -> canonical name, for building an
    `OwnerCandidate`."""

    by_key: dict[str, str] = field(default_factory=dict)
    name_by_id: dict[str, str] = field(default_factory=dict)


def load_internal_people_index(conn: psycopg.Connection) -> InternalPeopleIndex:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id, canonical_name, email FROM registry_entities "
            "WHERE kind = 'internal_person' AND crm_id IS NOT NULL AND active"
        )
        rows = cur.fetchall()
    index = InternalPeopleIndex()
    for crm_id, name, email in rows:
        index.name_by_id[crm_id] = name
        index.by_key[name.strip()] = crm_id
        if email:
            index.by_key[email.strip().lower()] = crm_id
    return index


def account_note_slugs(conn: psycopg.Connection) -> dict[str, str]:
    """`{crm_id: note_slug}` for every active account entity that already
    has a note (`registry_entities.note_slug` set); an account with no note
    has no signal notes either, so it is simply absent."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id, note_slug FROM registry_entities "
            "WHERE kind = 'account' AND active AND note_slug IS NOT NULL"
        )
        return dict(cur.fetchall())


def evidence_from_claims(
    conn: psycopg.Connection, account_note_slug: str, people: InternalPeopleIndex
) -> list[OwnerCandidate]:
    """One account's ranked evidence from the live graph: every unarchived
    claim on a `signal-*` note body-linked from the account's own note. A
    vendor rep "appears in" a claim's source when they are one of its
    `participants` (name or email match) or its `speaker`; "commitments
    they made for this account" is narrower -- `fact_type = 'commitment'`
    claims where they are the `speaker`.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_slug FROM links WHERE from_slug = %s AND kind = 'body' "
            "AND to_slug LIKE 'signal-%%'",
            (account_note_slug,),
        )
        signal_slugs = [row[0] for row in cur.fetchall()]
    if not signal_slugs:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.speaker, c.fact_type, c.source_slug, s.source_date, s.participants
            FROM claims c
            JOIN sources s ON s.slug = c.source_slug
            WHERE c.note_slug = ANY(%s) AND c.archived = false
            """,
            (signal_slugs,),
        )
        rows = cur.fetchall()

    sources_by_person: dict[str, set[str]] = {}
    last_seen_by_person: dict[str, date] = {}
    commitments_by_person: dict[str, int] = {}

    for speaker, fact_type, source_slug, source_date, participants in rows:
        present: set[str] = set()
        for participant in participants or []:
            name = (participant.get("name") or "").strip()
            crm_id = people.by_key.get(name)
            if crm_id is None:
                email = (participant.get("email") or "").strip().lower()
                crm_id = people.by_key.get(email) if email else None
            if crm_id:
                present.add(crm_id)
        speaker_id = people.by_key.get((speaker or "").strip())
        if speaker_id:
            present.add(speaker_id)

        for person_id in present:
            sources_by_person.setdefault(person_id, set()).add(source_slug)
            if source_date is not None and (
                person_id not in last_seen_by_person or source_date > last_seen_by_person[person_id]
            ):
                last_seen_by_person[person_id] = source_date
        if fact_type == "commitment" and speaker_id:
            commitments_by_person[speaker_id] = commitments_by_person.get(speaker_id, 0) + 1

    candidates = [
        OwnerCandidate(
            person_crm_id=person_id,
            name=people.name_by_id.get(person_id, person_id),
            source_count=len(sources),
            last_seen=last_seen_by_person.get(person_id),
            commitments=commitments_by_person.get(person_id, 0),
        )
        for person_id, sources in sources_by_person.items()
    ]
    return rank_candidates(candidates)


@dataclass
class ProposalRow:
    account_crm_id: str
    account_name: str
    candidates: list[OwnerCandidate]


def propose_owners(conn: psycopg.Connection) -> list[ProposalRow]:
    """Every active account with at least one ranked candidate (the
    real-customer path); an account with zero evidence gets no proposal.
    Writes nothing: `cli._cmd_propose_owners` turns each row into a
    `review_queue` insert."""
    people = load_internal_people_index(conn)
    slugs = account_note_slugs(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id, canonical_name FROM registry_entities "
            "WHERE kind = 'account' AND active AND crm_id IS NOT NULL "
            "ORDER BY canonical_name"
        )
        accounts = cur.fetchall()

    proposals: list[ProposalRow] = []
    for crm_id, name in accounts:
        note_slug = slugs.get(crm_id)
        if not note_slug:
            continue
        candidates = evidence_from_claims(conn, note_slug, people)
        if not candidates:
            continue
        proposals.append(
            ProposalRow(account_crm_id=crm_id, account_name=name, candidates=candidates)
        )
    return proposals


def confirm_owner_proposal(
    conn: psycopg.Connection, *, account_crm_id: str, owner_crm_id: str, commit: bool = True
) -> bool:
    """Set `registry_entities.owner_entity_id` for `account_crm_id` to the
    registry row named by `owner_crm_id`: Confirm on an `owner_proposal`
    review row.

    Refuses (returns `False`, no write) when `owner_crm_id` is not a
    vendor-side (`internal_person`) registry entity, or `account_crm_id` is
    not an `account` entity, so a corrupted or hand-edited payload can
    never make a customer contact an owner. On success, ensures the owner
    has a note, so the account's `owner:` link is never dead.

    Pass `commit=False` from inside a `with conn.transaction():` block,
    where psycopg refuses a manual `conn.commit()`."""
    if not is_vendor_side_crm_id(conn, owner_crm_id):
        return False
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM registry_entities WHERE crm_id = %s AND kind = 'account'",
            (account_crm_id,),
        )
        if cur.fetchone() is None:
            return False
        cur.execute(
            "UPDATE registry_entities SET owner_entity_id = "
            "(SELECT id FROM registry_entities WHERE crm_id = %s) "
            "WHERE crm_id = %s AND kind = 'account'",
            (owner_crm_id, account_crm_id),
        )
        updated = cur.rowcount > 0
    if updated:
        from portolan.notes.persist import ensure_entity_note

        registry = Registry.load(conn)
        ensure_entity_note(conn, registry, owner_crm_id, commit=commit)
    if commit:
        conn.commit()
    return updated


def backfill_action_owners(conn: psycopg.Connection) -> int:
    """Set `owner_entity_id` on every existing action row that has none,
    from its account's confirmed owner. The account is the entity note the
    action is body-linked from: that note itself when it is an `account`,
    the contact's `parent_id` when it is a `contact`; a product- or
    competitor-anchored action has no account and is left alone.

    Returns the number of action rows updated.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.id, n.type, re.crm_id, parent_re.crm_id
            FROM actions a
            JOIN links l ON l.to_slug = a.note_slug AND l.kind = 'body'
            JOIN notes n ON n.slug = l.from_slug
            LEFT JOIN registry_entities re ON re.note_slug = n.slug
            LEFT JOIN registry_entities parent_re ON parent_re.id = re.parent_id
            WHERE a.owner_entity_id IS NULL
            """
        )
        rows = cur.fetchall()

    updated = 0
    with conn.cursor() as cur:
        for action_id, note_type, own_crm_id, parent_crm_id in rows:
            if note_type == "account":
                account_crm_id = own_crm_id
            elif note_type == "contact":
                account_crm_id = parent_crm_id
            else:
                continue
            if not account_crm_id:
                continue
            cur.execute(
                "SELECT owner_entity_id FROM registry_entities "
                "WHERE crm_id = %s AND kind = 'account'",
                (account_crm_id,),
            )
            owner_row = cur.fetchone()
            if not owner_row or owner_row[0] is None:
                continue
            cur.execute(
                "UPDATE actions SET owner_entity_id = %s WHERE id = %s AND owner_entity_id IS NULL",
                (owner_row[0], action_id),
            )
            updated += cur.rowcount
    conn.commit()
    return updated
