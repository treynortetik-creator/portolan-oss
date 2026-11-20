"""Add a `branch` column to `corpus/out/crm/accounts.csv` from the truth
table's `branches` map (`01-spec/actions-and-owners-2026-09-19.md`, "Phase
2, revised contract": "Branch comes from a new `branch` attribute on
facility accounts loaded from the CRM export ... no reseed, load it like
region").

One-off, like `owner_crm_id` before it (that column was hand-added to this
same CSV the same way, ahead of a migration -- this file is not part of
the corpus generator's own `write_accounts_csv`, on purpose: adding it
there would only take effect on the NEXT full corpus regeneration, and the
spec is explicit that Phase 2 gets no reseed).

`corpus/truth-table.yaml`'s `meta.branches` is `[{id, name, serves: [account
id, ...]}, ...]` -- `serves` names truth-table account ids (`larkspur`,
`cobalt-ridge`, ...), a mix of systems and standalone facilities. Per the
spec ("branch attribute on FACILITY accounts"), a branch only ever gets
written onto a `kind == facility` row: a served id that is itself a
facility gets the branch directly; a served id that is a system gets the
branch on every facility whose `parent` is that system. A system with no
facility children (`st-aldric`, `sable-valley` in this corpus -- both are
served by a branch but have no child facility row) ends up with no row
carrying that branch at all, which is correct, not a bug: there is no
facility account to attach it to.

Idempotent: safe to re-run (rewrites the `branch` column from scratch
every time; every other column is carried through unchanged).
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

TRUTH_TABLE = Path(__file__).resolve().parent.parent / "corpus" / "truth-table.yaml"
ACCOUNTS_CSV = Path(__file__).resolve().parent.parent / "corpus" / "out" / "crm" / "accounts.csv"


def compute_branch_by_crm_id(truth: dict) -> dict[str, str]:
    accounts = truth["accounts"]
    by_id = {a["id"]: a for a in accounts}
    facility_children: dict[str, list[dict]] = {}
    for a in accounts:
        parent = a.get("parent")
        if parent:
            facility_children.setdefault(parent, []).append(a)

    branch_by_crm_id: dict[str, str] = {}
    for branch in truth["meta"]["branches"]:
        branch_id = branch["id"]
        for served_id in branch["serves"]:
            served = by_id.get(served_id)
            if served is None:
                continue
            if served.get("kind") == "facility":
                branch_by_crm_id[served["crm_id"]] = branch_id
                continue
            for child in facility_children.get(served_id, []):
                if child.get("kind") == "facility":
                    branch_by_crm_id[child["crm_id"]] = branch_id
    return branch_by_crm_id


def main() -> None:
    with TRUTH_TABLE.open(encoding="utf-8") as fh:
        truth = yaml.safe_load(fh)
    branch_by_crm_id = compute_branch_by_crm_id(truth)

    with ACCOUNTS_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())
    if "branch" not in fieldnames:
        fieldnames.append("branch")
    for row in rows:
        row["branch"] = branch_by_crm_id.get(row["crm_id"], "")

    with ACCOUNTS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    assigned = sum(1 for v in branch_by_crm_id.values() if v)
    print(f"branch column written: {assigned} account(s) assigned a branch, {len(rows)} rows total")


if __name__ == "__main__":
    main()
