"""`Registry.import_crm()`'s implementation: CSV parsing and the dedup
rules from `portolan-spec-v2.md` sec 3 / `open-questions.md` sec F "B2.
Dirty CRMs".

Dedup rules, verbatim from the spec:
    - same email domain + same normalised name -> merge
    - same normalised name, different domain -> `review_queue` row, kind
      `ambiguous_entity`, keep both
    - contacts with the same email -> merge
Account names compare by `text.light_normalize`, which keeps legal suffixes.

Id scheme (`registry.py`'s module docstring): `registry_entities.crm_id` is
the external string entity id. Every row gets one, real (from the CRM
export) or synthesised (`text.slugify` of the name) when the export leaves
it blank.

Per-account aliases arrive two ways: an `aliases` CSV column
(semicolon-separated), or `account_aliases`, keyed by `crm_id`, which the
CLI (`cli.py`) builds from a truth-table-shaped `--meta` file's
`accounts:` list (`corpus/out/crm/accounts.csv` has no alias column).
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from portolan.registry.models import ImportReport
from portolan.registry.text import (
    acronym_of,
    domain_of_email,
    light_normalize,
    slugify,
    strip_legal_suffix,
)

# registry_aliases.source is a closed CHECK ('crm', 'tracker', 'human', 'extractor');
# every alias this module writes is derived at CRM-import time, so all are 'crm'.
_ALIAS_SOURCE = "crm"


def _read_csv(path: Path | str) -> list[dict[str, str]]:
    with open(Path(path), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _is_inactive(status: str | None) -> bool:
    if not status:
        return False
    lowered = status.lower()
    return "churn" in lowered or "left" in lowered


_STATUS_DATE_RE = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?")


def _inactive_since(status: str | None) -> date | None:
    """The date an inactive status names (`left_2026-07` -> 2026-07-01,
    `churned_2026-02-15` -> 2026-02-15); `None` for an active status or
    one with no date in it."""
    if not _is_inactive(status):
        return None
    match = _STATUS_DATE_RE.search(status or "")
    if match is None:
        return None
    year, month, day = match.group(1), match.group(2), match.group(3)
    try:
        return date(int(year), int(month), int(day or 1))
    except ValueError:
        return None


# Team names a source uses for the vendor itself ("Halvard Support" on a
# ticket reply). Seeded as aliases of the vendor's own registry entity so
# they resolve instead of landing in the review queue as unknown names.
_VENDOR_TEAM_WORDS: tuple[str, ...] = (
    "Support",
    "Support Team",
    "Customer Support",
    "Customer Success",
    "Operations",
    "Logistics",
    "Sales",
    "Team",
)


def _vendor_aliases(name: str) -> list[str]:
    """The vendor's short name (its first word, when the name has more
    than one and the first is longer than three letters) and every
    `<name or short name> <team word>` form."""
    words = name.split()
    bases = [name]
    if len(words) > 1 and len(words[0]) > 3:
        bases.append(words[0])
    aliases = [b for b in bases if b != name]
    aliases.extend(f"{base} {team}" for base in bases for team in _VENDOR_TEAM_WORDS)
    return aliases


def _branch_aliases(name: str) -> list[str]:
    """`Dayton distribution branch` -> `Dayton`, `Dayton branch`."""
    words = name.split()
    if len(words) < 2:
        return []
    return [words[0], f"{words[0]} branch"]


def _no_aliases(name: str) -> list[str]:
    return []


def _unique_id(base: str, taken: dict[str, int]) -> str:
    if base not in taken:
        return base
    n = 2
    candidate = f"{base}-{n}"
    while candidate in taken:
        n += 1
        candidate = f"{base}-{n}"
    return candidate


def _add_alias_row(cur, dbid: int, alias: str, *, source: str = _ALIAS_SOURCE) -> None:
    alias = alias.strip()
    if not alias:
        return
    cur.execute(
        "INSERT INTO registry_aliases (alias, entity_id, source) VALUES (%s, %s, %s) "
        "ON CONFLICT (alias, entity_id) DO NOTHING",
        (alias, dbid, source),
    )


def _write_review_queue(cur, *, kind: str, payload: dict[str, Any]) -> None:
    cur.execute(
        "INSERT INTO review_queue (kind, payload) VALUES (%s, %s::jsonb)",
        (kind, json.dumps(payload)),
    )


def _fetch_by_kind(cur, kind: str) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT id, crm_id, canonical_name, email, email_domain
        FROM registry_entities WHERE kind = %s
        """,
        (kind,),
    )
    return [
        {"id": r[0], "crm_id": r[1], "canonical_name": r[2], "email": r[3], "email_domain": r[4]}
        for r in cur.fetchall()
    ]


def _preload_crm_ids(cur) -> dict[str, int]:
    cur.execute("SELECT crm_id, id FROM registry_entities WHERE crm_id IS NOT NULL")
    return dict(cur.fetchall())


def _import_accounts(
    cur,
    accounts_csv: Path | str,
    crm_id_to_dbid: dict[str, int],
    report: ImportReport,
) -> list[tuple[str, str]]:
    """Insert/update/merge every account row. Returns `[(final_crm_id,
    name), ...]` in CSV order — merged-away rows resolve to the surviving
    entity's id — for the later parent-linking and alias-seeding passes."""
    rows = _read_csv(accounts_csv)
    existing = _fetch_by_kind(cur, "account")
    processed: list[tuple[str, str]] = []

    for row in rows:
        crm_id = _clean(row.get("crm_id"))
        name = (row.get("name") or "").strip()
        kind_col = _clean(row.get("kind"))
        domain = (_clean(row.get("domain")) or "").lower() or None
        active = not _is_inactive(row.get("status"))
        inactive_since = _inactive_since(row.get("status"))
        dedupe_key = light_normalize(name)

        if crm_id and crm_id in crm_id_to_dbid:
            dbid = crm_id_to_dbid[crm_id]
            cur.execute(
                "UPDATE registry_entities SET canonical_name=%s, account_kind=%s, "
                "email_domain=%s, active=%s, inactive_since=%s, synced_at=now() WHERE id=%s",
                (name, kind_col, domain, active, inactive_since, dbid),
            )
            report.updated += 1
            processed.append((crm_id, name))
            continue

        dupe = next(
            (a for a in existing if light_normalize(a["canonical_name"]) == dedupe_key), None
        )
        if dupe is not None:
            if (dupe["email_domain"] or None) == domain:
                report.merged += 1
                report.notes.append(
                    f"merged account {name!r} (crm_id={crm_id!r}) into existing "
                    f"crm_id={dupe['crm_id']!r} — same normalised name, same email domain"
                )
                if crm_id:
                    crm_id_to_dbid[crm_id] = dupe["id"]
                    _add_alias_row(cur, dupe["id"], crm_id)
                processed.append((dupe["crm_id"], name))
                continue

            final_id = crm_id or _unique_id(slugify(name), crm_id_to_dbid)
            dbid = _insert_account(cur, final_id, name, kind_col, domain, active, inactive_since)
            crm_id_to_dbid[final_id] = dbid
            existing.append(
                {"id": dbid, "crm_id": final_id, "canonical_name": name, "email_domain": domain}
            )
            report.created += 1
            _write_review_queue(
                cur,
                kind="ambiguous_entity",
                payload={
                    "mention": name,
                    "reason": "same normalised name, different email domain",
                    "candidates": [dupe["crm_id"], final_id],
                },
            )
            report.queued += 1
            processed.append((final_id, name))
            continue

        final_id = crm_id or _unique_id(slugify(name), crm_id_to_dbid)
        dbid = _insert_account(cur, final_id, name, kind_col, domain, active, inactive_since)
        crm_id_to_dbid[final_id] = dbid
        existing.append(
            {"id": dbid, "crm_id": final_id, "canonical_name": name, "email_domain": domain}
        )
        report.created += 1
        processed.append((final_id, name))

    # Second pass: parent linking, now every crm_id in this file maps to a
    # db id (real, merged-into, or synthesised). A link that would make an
    # account its own ancestor is left out, so every walk up the chain ends.
    cur.execute("SELECT id, parent_id FROM registry_entities WHERE kind = 'account'")
    parent_of: dict[int, int | None] = dict(cur.fetchall())
    for row in rows:
        crm_id = _clean(row.get("crm_id"))
        parent_crm_id = _clean(row.get("parent_crm_id"))
        if not parent_crm_id:
            continue
        child_dbid = crm_id_to_dbid.get(crm_id) if crm_id else None
        parent_dbid = crm_id_to_dbid.get(parent_crm_id)
        if not (child_dbid and parent_dbid):
            continue
        if _would_loop(parent_of, child_dbid, parent_dbid):
            report.notes.append(
                f"parent link {crm_id!r} -> {parent_crm_id!r} left out: it would make "
                f"{crm_id!r} its own ancestor"
            )
            continue
        cur.execute(
            "UPDATE registry_entities SET parent_id=%s WHERE id=%s", (parent_dbid, child_dbid)
        )
        parent_of[child_dbid] = parent_dbid

    return processed


def _would_loop(parent_of: dict[int, int | None], child: int, parent: int) -> bool:
    """Whether making `parent` the parent of `child` puts `child` on its
    own ancestor chain."""
    seen: set[int] = set()
    node: int | None = parent
    while node is not None and node not in seen:
        if node == child:
            return True
        seen.add(node)
        node = parent_of.get(node)
    return False


def _insert_account(
    cur,
    crm_id: str,
    name: str,
    kind_col: str | None,
    domain: str | None,
    active: bool,
    inactive_since: date | None,
) -> int:
    cur.execute(
        """
        INSERT INTO registry_entities
            (kind, canonical_name, crm_id, account_kind, email_domain, active,
             inactive_since, synced_at)
        VALUES ('account', %s, %s, %s, %s, %s, %s, now())
        RETURNING id
        """,
        (name, crm_id, kind_col, domain, active, inactive_since),
    )
    return cur.fetchone()[0]


def _import_contacts(
    cur, contacts_csv: Path | str, crm_id_to_dbid: dict[str, int], report: ImportReport
) -> None:
    rows = _read_csv(contacts_csv)
    existing = _fetch_by_kind(cur, "contact")

    for row in rows:
        crm_id = _clean(row.get("crm_id"))
        name = (row.get("name") or "").strip()
        email = (_clean(row.get("email")) or "").lower() or None
        account_crm_id = _clean(row.get("account_crm_id"))
        active = not _is_inactive(row.get("status"))
        inactive_since = _inactive_since(row.get("status"))
        parent_dbid = crm_id_to_dbid.get(account_crm_id) if account_crm_id else None
        domain = domain_of_email(email)

        if crm_id and crm_id in crm_id_to_dbid:
            dbid = crm_id_to_dbid[crm_id]
            cur.execute(
                "UPDATE registry_entities SET canonical_name=%s, email=%s, email_domain=%s, "
                "parent_id=%s, active=%s, inactive_since=%s, synced_at=now() WHERE id=%s",
                (name, email, domain, parent_dbid, active, inactive_since, dbid),
            )
            report.updated += 1
            continue

        dupe = next((c for c in existing if email and c["email"] == email), None)
        if dupe is not None:
            report.merged += 1
            report.notes.append(
                f"merged contact {name!r} (crm_id={crm_id!r}) into existing "
                f"crm_id={dupe['crm_id']!r} — same email"
            )
            if crm_id:
                crm_id_to_dbid[crm_id] = dupe["id"]
                _add_alias_row(cur, dupe["id"], crm_id)
            continue

        final_id = crm_id or _unique_id(slugify(name), crm_id_to_dbid)
        cur.execute(
            """
            INSERT INTO registry_entities
                (kind, canonical_name, crm_id, email, email_domain, parent_id, active,
                 inactive_since, synced_at)
            VALUES ('contact', %s, %s, %s, %s, %s, %s, %s, now())
            RETURNING id
            """,
            (name, final_id, email, domain, parent_dbid, active, inactive_since),
        )
        dbid = cur.fetchone()[0]
        crm_id_to_dbid[final_id] = dbid
        existing.append({"id": dbid, "crm_id": final_id, "canonical_name": name, "email": email})
        report.created += 1


def _import_meta_entities(
    cur,
    kind: str,
    entries: list[dict[str, Any]],
    crm_id_to_dbid: dict[str, int],
    report: ImportReport,
    derived_aliases: Callable[[str], list[str]] = _no_aliases,
) -> None:
    for entry in entries:
        entry_id = entry.get("id") or slugify(entry.get("name", kind))
        name = entry.get("name") or entry_id
        email = (entry.get("email") or "").strip().lower() or None
        domain = domain_of_email(email)

        if entry_id in crm_id_to_dbid:
            dbid = crm_id_to_dbid[entry_id]
            cur.execute(
                "UPDATE registry_entities SET canonical_name=%s, email=%s, email_domain=%s, "
                "active=true, synced_at=now() WHERE id=%s",
                (name, email, domain, dbid),
            )
            report.updated += 1
        else:
            cur.execute(
                """
                INSERT INTO registry_entities
                    (kind, canonical_name, crm_id, email, email_domain, active, synced_at)
                VALUES (%s, %s, %s, %s, %s, true, now())
                RETURNING id
                """,
                (kind, name, entry_id, email, domain),
            )
            dbid = cur.fetchone()[0]
            crm_id_to_dbid[entry_id] = dbid
            report.created += 1

        for alias in [*(entry.get("aliases") or []), *derived_aliases(name)]:
            _add_alias_row(cur, dbid, alias)


def _seed_account_aliases(
    cur,
    processed: list[tuple[str, str]],
    crm_id_to_dbid: dict[str, int],
    account_aliases: dict[str, list[str]],
    csv_aliases_by_crm_id: dict[str, list[str]],
) -> None:
    """Every account gets aliases from its aliases lists, its name without
    legal suffixes, and its acronym when the name is 2 to 4 capitalised words."""
    seen_ids: set[str] = set()
    for entity_id, name in processed:
        if entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)
        dbid = crm_id_to_dbid.get(entity_id)
        if not dbid:
            continue

        for alias in account_aliases.get(entity_id, []):
            _add_alias_row(cur, dbid, alias)
        for alias in csv_aliases_by_crm_id.get(entity_id, []):
            _add_alias_row(cur, dbid, alias)

        stripped = strip_legal_suffix(name)
        if stripped and stripped != name:
            _add_alias_row(cur, dbid, stripped)

        acronym = acronym_of(name)
        if acronym:
            _add_alias_row(cur, dbid, acronym)


def run_import_crm(
    conn,
    accounts_csv: Path | str,
    contacts_csv: Path | str,
    *,
    products: list[dict[str, Any]] | None = None,
    competitors: list[dict[str, Any]] | None = None,
    internal_people: list[dict[str, Any]] | None = None,
    account_aliases: dict[str, list[str]] | None = None,
    supplier: dict[str, Any] | None = None,
    branches: list[dict[str, Any]] | None = None,
) -> ImportReport:
    """`supplier` and `branches` (the meta block's own `supplier:` and
    `branches:`) are the vendor itself and its sites. Both import as kind
    `other` with no email domain; they exist so a mention of the vendor or
    a branch resolves instead of being queued as an unknown name."""
    report = ImportReport()
    account_aliases = account_aliases or {}

    with conn.cursor() as cur:
        crm_id_to_dbid = _preload_crm_ids(cur)

        accounts_rows = _read_csv(accounts_csv)
        csv_aliases_by_crm_id: dict[str, list[str]] = {}
        for row in accounts_rows:
            crm_id = _clean(row.get("crm_id"))
            raw_aliases = _clean(row.get("aliases"))
            if crm_id and raw_aliases:
                csv_aliases_by_crm_id[crm_id] = [
                    a.strip() for a in raw_aliases.split(";") if a.strip()
                ]

        processed = _import_accounts(cur, accounts_csv, crm_id_to_dbid, report)
        _import_contacts(cur, contacts_csv, crm_id_to_dbid, report)

        for kind, entries in (
            ("product", products),
            ("competitor", competitors),
            ("internal_person", internal_people),
        ):
            if entries:
                _import_meta_entities(cur, kind, entries, crm_id_to_dbid, report)
        if supplier and supplier.get("name"):
            vendor = {
                "id": supplier.get("id"),
                "name": supplier["name"],
                "aliases": list(supplier.get("aliases") or []),
            }
            _import_meta_entities(
                cur, "other", [vendor], crm_id_to_dbid, report, derived_aliases=_vendor_aliases
            )
        if branches:
            sites = [{"id": b.get("id"), "name": b["name"]} for b in branches if b.get("name")]
            _import_meta_entities(
                cur, "other", sites, crm_id_to_dbid, report, derived_aliases=_branch_aliases
            )

        _seed_account_aliases(
            cur, processed, crm_id_to_dbid, account_aliases, csv_aliases_by_crm_id
        )

    conn.commit()
    return report
