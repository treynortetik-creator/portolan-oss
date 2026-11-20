"""Tests for `scripts/set_owners.py` (`01-spec/actions-and-owners-
2026-09-19.md`, Phase 1, build item 2: "a script ... that takes the
assignment CSV and sets registry_entities.owner_entity_id idempotently,
printing counts only").

`scripts/` isn't a Python package (no `__init__.py`, matching every other
file there -- `pyproject.toml`'s own per-file ruff ignore treats them as
standalone tools, not modules the test suite imports normally), so this
loads the module by file path.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

from portolan.registry import Registry
from tests._paths import require_repo_path
from tests.merge_helpers import (
    ASHGROVE,
    SAM_OKAFOR,
    loaded_registry_fixture,  # noqa: F401
)

SCRIPT_PATH = require_repo_path("scripts", "set_owners.py")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_set_owners():
    spec = importlib.util.spec_from_file_location("set_owners_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["set_owners_script"] = module
    spec.loader.exec_module(module)
    return module.set_owners


def _write_accounts_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "crm_id",
        "name",
        "kind",
        "parent_crm_id",
        "domain",
        "arr",
        "region",
        "status",
        "owner_crm_id",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**dict.fromkeys(fieldnames, ""), **row})


def test_set_owners_updates_and_is_idempotent(loaded_registry: Registry, db, tmp_path):
    set_owners = _load_set_owners()
    csv_path = tmp_path / "accounts.csv"
    _write_accounts_csv(
        csv_path, [{"crm_id": ASHGROVE, "name": "Ashgrove Health", "owner_crm_id": SAM_OKAFOR}]
    )

    first = set_owners(db, csv_path)
    assert first == {
        "total_rows": 1,
        "updated": 1,
        "already_correct": 0,
        "skipped_no_owner_field": 0,
        "skipped_account_not_found": 0,
        "skipped_owner_not_vendor_side": 0,
    }

    with db.cursor() as cur:
        cur.execute(
            "SELECT owner.crm_id FROM registry_entities a "
            "JOIN registry_entities owner ON owner.id = a.owner_entity_id "
            "WHERE a.crm_id = %s",
            (ASHGROVE,),
        )
        assert cur.fetchone() == (SAM_OKAFOR,)

    second = set_owners(db, csv_path)
    assert second["updated"] == 0
    assert second["already_correct"] == 1


def test_set_owners_skips_blank_owner_and_unknown_account(loaded_registry: Registry, db, tmp_path):
    set_owners = _load_set_owners()
    csv_path = tmp_path / "accounts.csv"
    _write_accounts_csv(
        csv_path,
        [
            {"crm_id": ASHGROVE, "name": "Ashgrove Health", "owner_crm_id": ""},
            {"crm_id": "A-9999-NOT-REAL", "name": "Nowhere", "owner_crm_id": SAM_OKAFOR},
        ],
    )

    counts = set_owners(db, csv_path)

    assert counts["updated"] == 0
    assert counts["skipped_no_owner_field"] == 1
    assert counts["skipped_account_not_found"] == 1


def test_set_owners_refuses_a_non_vendor_side_owner(loaded_registry: Registry, db, tmp_path):
    set_owners = _load_set_owners()
    with db.cursor() as cur:
        cur.execute("SELECT crm_id FROM registry_entities WHERE kind = 'contact' LIMIT 1")
        customer_contact_crm_id = cur.fetchone()[0]

    csv_path = tmp_path / "accounts.csv"
    _write_accounts_csv(
        csv_path,
        [{"crm_id": ASHGROVE, "name": "Ashgrove Health", "owner_crm_id": customer_contact_crm_id}],
    )

    counts = set_owners(db, csv_path)

    assert counts["updated"] == 0
    assert counts["skipped_owner_not_vendor_side"] == 1
    with db.cursor() as cur:
        cur.execute("SELECT owner_entity_id FROM registry_entities WHERE crm_id = %s", (ASHGROVE,))
        assert cur.fetchone()[0] is None
