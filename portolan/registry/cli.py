"""`python -m portolan.registry import --accounts ... --contacts ... [--meta ...]`

`--meta` points at a YAML file shaped like `corpus/truth-table.yaml`: a
top-level `meta:` block (`products`, `competitors`, `internal_people`,
`supplier` -- the vendor itself -- and its `branches`) plus a top-level
`accounts:` list, each entry optionally carrying an `aliases:` list keyed
by `crm_id`. The meta block seeds `Registry.import_crm`'s keyword
parameters directly; the accounts list becomes `account_aliases`
(`importer.py`'s module docstring).

`--meta` is optional: without it the import seeds no products,
competitors, internal people or extra account aliases -- the shape of a
customer's own CRM export with no truth table alongside it.

Other subcommands: `assign-owners`, `propose-owners`,
`backfill-action-owners` (`owners.py`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from portolan import db
from portolan.registry.owners import (
    assign_owners_from_corpus,
    backfill_action_owners,
    propose_owners,
    write_owner_crm_id_column,
)
from portolan.registry.registry import Registry

_MetaTuple = tuple[list | None, list | None, list | None, dict[str, list[str]]]


def _load_meta(meta_path: Path | None) -> _MetaTuple:
    if meta_path is None:
        return None, None, None, {}

    data = yaml.safe_load(meta_path.read_text()) or {}
    meta = data.get("meta") or {}
    products = meta.get("products")
    competitors = meta.get("competitors")
    internal_people = meta.get("internal_people")

    account_aliases: dict[str, list[str]] = {}
    for account in data.get("accounts") or []:
        crm_id = account.get("crm_id")
        aliases = account.get("aliases")
        if crm_id and aliases:
            account_aliases[crm_id] = list(aliases)

    return products, competitors, internal_people, account_aliases


def _load_org_meta(meta_path: Path | None) -> tuple[dict | None, list | None]:
    """The meta block's `supplier` (the vendor itself) and `branches`."""
    if meta_path is None:
        return None, None
    meta = (yaml.safe_load(meta_path.read_text()) or {}).get("meta") or {}
    return meta.get("supplier"), meta.get("branches")


def _cmd_import(args: argparse.Namespace) -> None:
    products, competitors, internal_people, account_aliases = _load_meta(args.meta)
    supplier, branches = _load_org_meta(args.meta)

    conn = db.connect()
    try:
        report = Registry().import_crm(
            conn,
            args.accounts,
            args.contacts,
            products=products,
            competitors=competitors,
            internal_people=internal_people,
            account_aliases=account_aliases or None,
            supplier=supplier,
            branches=branches,
        )
    finally:
        conn.close()

    print(
        f"created={report.created} updated={report.updated} "
        f"merged={report.merged} queued={report.queued}"
    )
    for note in report.notes:
        print(f"  - {note}")


def _cmd_assign_owners(args: argparse.Namespace) -> None:
    """`python -m portolan.registry assign-owners --corpus corpus/out
    [--write-csv]` (`01-spec/actions-and-owners-2026-09-19.md`, "Assignment
    rule for the synthetic world"). Reads `<corpus>/manifest.json` and the
    sibling `truth-table.yaml` (`corpus/truth-table.yaml` next to
    `corpus/out/`) -- no database connection, so it can run on a freshly
    generated corpus before anything is seeded.
    """
    from portolan.bench.truth import Manifest, TruthTable

    corpus_dir: Path = args.corpus
    manifest_path = corpus_dir / "manifest.json"
    truth_table_path = corpus_dir.parent / "truth-table.yaml"
    table = TruthTable.load(truth_table_path)
    manifest = Manifest.load(manifest_path)

    assignments = assign_owners_from_corpus(table, manifest)

    header = (
        f"{'account':35s} {'owner':18s} {'sources':>7s} {'last seen':>10s} "
        f"{'commitments':>11s}  basis"
    )
    print(header)
    print("-" * len(header))
    for account in table.accounts:
        r = assignments[account.id]
        last_seen = r.last_seen.isoformat() if r.last_seen else "-"
        print(
            f"{r.account_name:35s} {r.owner_name:18s} {r.source_count:>7d} "
            f"{last_seen:>10s} {r.commitments:>11d}  {r.basis}"
        )

    if args.write_csv:
        accounts_csv = corpus_dir / "crm" / "accounts.csv"
        write_owner_crm_id_column(accounts_csv, assignments)
        print(f"\nwrote owner_crm_id column to {accounts_csv}")


def _cmd_propose_owners(args: argparse.Namespace) -> None:
    """`python -m portolan.registry propose-owners`: one `review_queue` row
    (kind `owner_proposal`) per active account with at least one ranked
    candidate. Safe to rerun: an account that already has an open proposal
    is skipped, not duplicated."""
    conn = db.connect()
    try:
        proposals = propose_owners(conn)
        created = 0
        skipped = 0
        with conn.cursor() as cur:
            for proposal in proposals:
                cur.execute(
                    "SELECT 1 FROM review_queue WHERE kind = 'owner_proposal' "
                    "AND status = 'open' AND payload ->> 'account_crm_id' = %s",
                    (proposal.account_crm_id,),
                )
                if cur.fetchone() is not None:
                    skipped += 1
                    continue
                payload = {
                    "account_crm_id": proposal.account_crm_id,
                    "account_name": proposal.account_name,
                    "candidates": [
                        {
                            "crm_id": c.person_crm_id,
                            "name": c.name,
                            "source_count": c.source_count,
                            "last_seen": c.last_seen.isoformat() if c.last_seen else None,
                            "commitments": c.commitments,
                        }
                        for c in proposal.candidates
                    ],
                }
                cur.execute(
                    "INSERT INTO review_queue (kind, payload) VALUES ('owner_proposal', %s::jsonb)",
                    (json.dumps(payload),),
                )
                created += 1
        conn.commit()
    finally:
        conn.close()
    print(f"created={created} skipped(already open)={skipped} accounts_considered={len(proposals)}")


def _cmd_backfill_action_owners(args: argparse.Namespace) -> None:
    """`python -m portolan.registry backfill-action-owners`: set
    `owner_entity_id` on every existing action row that has none, from the
    account it belongs to."""
    conn = db.connect()
    try:
        updated = backfill_action_owners(conn)
    finally:
        conn.close()
    print(f"updated={updated}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m portolan.registry")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="import registry_entities from a CRM export")
    p_import.add_argument("--accounts", required=True, type=Path, help="accounts CSV")
    p_import.add_argument("--contacts", required=True, type=Path, help="contacts CSV")
    p_import.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="optional truth-table-shaped YAML seeding products/competitors/internal_people "
        "and per-account aliases",
    )
    p_import.set_defaults(func=_cmd_import)

    p_assign_owners = sub.add_parser(
        "assign-owners",
        help="assign account owners from corpus/out (the synthetic-world path, no DB)",
    )
    p_assign_owners.add_argument(
        "--corpus", required=True, type=Path, help="path to a corpus's out/ directory"
    )
    p_assign_owners.add_argument(
        "--write-csv",
        action="store_true",
        help="write the owner_crm_id column onto <corpus>/crm/accounts.csv",
    )
    p_assign_owners.set_defaults(func=_cmd_assign_owners)

    p_propose_owners = sub.add_parser(
        "propose-owners",
        help="write owner_proposal review rows from the live graph's own claims",
    )
    p_propose_owners.set_defaults(func=_cmd_propose_owners)

    p_backfill = sub.add_parser(
        "backfill-action-owners",
        help="set owner_entity_id on existing action rows from their account's owner",
    )
    p_backfill.set_defaults(func=_cmd_backfill_action_owners)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
