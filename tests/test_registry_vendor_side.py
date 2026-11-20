"""The registry holds the vendor itself and its branches (the meta block's
`supplier:` and `branches:`), and resolves a departed contact or churned
account for a fact said before the CRM's departure date (migration 0032).
`tests/test_review_queue_volume.py` covers the same rules end to end."""

from __future__ import annotations

from datetime import date

from portolan.extract.schema import EntityKind
from portolan.registry import Registry
from portolan.registry.cli import _load_meta, _load_org_meta
from portolan.registry.importer import _inactive_since
from tests.merge_helpers import ACCOUNTS_CSV, CONTACTS_CSV, TRUTH_TABLE


def _registry(db) -> Registry:
    products, competitors, internal_people, account_aliases = _load_meta(TRUTH_TABLE)
    supplier, branches = _load_org_meta(TRUTH_TABLE)
    Registry().import_crm(
        db,
        ACCOUNTS_CSV,
        CONTACTS_CSV,
        products=products,
        competitors=competitors,
        internal_people=internal_people,
        account_aliases=account_aliases,
        supplier=supplier,
        branches=branches,
    )
    return Registry.load(db)


def test_the_vendor_and_its_team_names_resolve_to_one_entity(db) -> None:
    registry = _registry(db)
    ids = {
        mention: registry.resolve(mention).entity_id
        for mention in ("Halvard Medical", "Halvard", "Halvard Support", "halvard support team")
    }
    assert set(ids.values()) == {"halvard-medical"}
    vendor = registry.get("halvard-medical")
    assert vendor is not None
    assert vendor.kind == EntityKind.OTHER
    # No domain on the vendor: an all-staff email stays "internal".
    assert vendor.email_domain is None


def test_a_branch_resolves_by_its_city(db) -> None:
    registry = _registry(db)
    assert registry.resolve("Columbus").entity_id == "columbus"
    assert registry.resolve("Dayton branch").entity_id == "dayton"


def test_a_departed_contact_resolves_only_for_facts_said_before_they_left(db) -> None:
    registry = _registry(db)  # Nina Vale: left_2026-08 in contacts.csv
    assert registry.resolve("Nina Vale").status == "unknown"
    assert registry.resolve("Nina Vale", as_of=date(2026, 3, 20)).entity_id == "C-2060"
    assert registry.resolve("Nina Vale", as_of=date(2026, 8, 1)).entity_id == "C-2060"
    assert registry.resolve("Nina Vale", as_of=date(2026, 8, 2)).status == "unknown"


def test_inactive_since_reads_the_status_date() -> None:
    assert _inactive_since("left_2026-07") == date(2026, 7, 1)
    assert _inactive_since("churned_2026-02-15") == date(2026, 2, 15)
    assert _inactive_since("churned") is None
    assert _inactive_since("added_2026-08") is None
    assert _inactive_since(None) is None
