"""CRM changes become claims through code (`portolan.crm`,
`01-spec/monday-email-2026-09-25.md`): snapshots per export date, a field
compared against the last export that carried it, one claim per change with
the sentence and every number computed here, cited to both CRM rows."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from portolan.connectors.file_connector import FileConnector
from portolan.crm.changes import Names, change_sentence, compute_changes
from portolan.crm.ingest import ingest_crm_snapshots
from portolan.crm.snapshots import Snapshot, snapshot_date
from portolan.crm.truth import CallClaim, is_negative, truth_flags
from portolan.registry import Registry
from portolan.store import LocalStore

OPP = "opportunity"


def _snap(sid: int, obj: str, crm_id: str, as_of: str, account: str | None, **fields: str):
    return Snapshot(
        id=sid,
        object=obj,
        crm_id=crm_id,
        account_crm_id=account,
        as_of=date.fromisoformat(as_of),
        export_key=f"crm/{obj}-{as_of}.csv",
        fields=dict(fields),
        source_slug=f"S-{sid}",
        locator=f"crm/{obj}-{as_of}.csv#{crm_id}",
    )


def test_each_field_is_compared_with_the_last_export_that_carried_it():
    snaps = [
        _snap(1, OPP, "O-1", "2026-04-15", "A-1", name="Acme renewal", stage="negotiation",
              close_date="2027-03-31", amount="1420000"),
        _snap(2, OPP, "O-1", "2026-08-14", "A-1", name="Acme renewal", stage="negotiation",
              close_date="2027-03-31", amount="1420000", health="yellow"),
        # An export without the health column says nothing about health.
        _snap(3, OPP, "O-1", "2026-08-21", "A-1", name="Acme renewal", stage="negotiation",
              close_date="2027-06-30", amount="1420000"),
        _snap(4, OPP, "O-1", "2026-08-28", "A-1", name="Acme renewal", stage="commit",
              close_date="2027-06-30", amount="1562000", health="green"),
    ]  # fmt: skip
    changes = {(c.field, c.as_of.isoformat()): c for c in compute_changes(snaps)}

    assert set(changes) == {
        ("close_date", "2026-08-21"),
        ("stage", "2026-08-28"),
        ("amount", "2026-08-28"),
        ("health", "2026-08-28"),
    }
    assert changes[("close_date", "2026-08-21")].prev.id == 2
    assert changes[("health", "2026-08-28")].prev.id == 2
    assert changes[("stage", "2026-08-28")].prev.id == 3

    names = Names({"A-1": "Acme Health"})
    assert change_sentence(changes[("amount", "2026-08-28")], names) == (
        "Acme renewal: amount rose from $1,420,000 to $1,562,000, up $142,000 or 10.0% "
        "(CRM, 2026-08-28)."
    )
    assert change_sentence(changes[("close_date", "2026-08-21")], names) == (
        "Acme renewal: close date moved from 2027-03-31 to 2027-06-30, 91 days later "
        "(CRM, 2026-08-21)."
    )
    assert change_sentence(changes[("stage", "2026-08-28")], names) == (
        "Acme renewal: stage moved from Negotiation to Commit (CRM, 2026-08-28)."
    )
    # Forward: stage toward a close, amount up, health up. A slipped close date is not.
    assert [changes[k].direction for k in sorted(changes)] == [1, -1, 1, 1]


def test_records_added_and_contacts_removed_but_not_from_a_partial_export():
    snaps = [
        _snap(1, "contact", "C-1", "2026-08-14", "A-1", name="Tom Ridley", title="Buyer"),
        _snap(2, "contact", "C-2", "2026-08-14", "A-1", name="Aisha Bell", title="Manager"),
        _snap(3, "contact", "C-9", "2026-08-14", "A-9", name="Nobody Else", title="CFO"),
        _snap(4, "contact", "C-2", "2026-08-28", "A-1", name="Aisha Bell", title="Manager"),
        _snap(5, "contact", "C-3", "2026-08-28", "A-1", name="Lena Ortiz", title="Nurse"),
        _snap(6, OPP, "O-1", "2026-08-14", "A-1", name="Old deal", stage="proposal"),
        _snap(7, OPP, "O-2", "2026-08-28", "A-1", name="New pilot", stage="qualification",
              amount="48000", close_date="2026-10-01"),
    ]  # fmt: skip
    found = {(c.kind, c.crm_id) for c in compute_changes(snaps)}

    # C-9's account is not in the newer export at all: a partial export, not a removal.
    assert found == {("removed", "C-1"), ("added", "C-3"), ("added", "O-2")}
    names = Names({"A-1": "Acme Health"})
    by_id = {c.crm_id: c for c in compute_changes(snaps)}
    assert change_sentence(by_id["O-2"], names) == (
        "Acme Health: new opportunity New pilot, at stage Qualification, amount $48,000, "
        "close date 2026-10-01 (CRM, 2026-08-28)."
    )
    assert change_sentence(by_id["C-1"], names) == (
        "Acme Health: contact removed, Tom Ridley, Buyer; not in the 2026-08-28 contacts "
        "export (CRM, 2026-08-28)."
    )


def test_the_export_date_comes_from_the_row_then_the_file_name_then_the_api_date():
    today = date(2026, 9, 25)
    assert snapshot_date({"export_date": "2026-08-21"}, {"path": "crm/x.csv"}, today) == date(
        2026, 8, 21
    )
    assert snapshot_date({}, {"path": "crm/contacts-2026-08-28.csv"}, today) == date(2026, 8, 28)
    # A file export with no date cannot be placed in the week sequence.
    assert snapshot_date({}, {"path": "crm/contacts.csv"}, today) is None
    assert snapshot_date({}, {}, today) == today


def test_a_forward_change_is_flagged_against_a_recent_negative_call_on_the_family():
    from portolan.crm.changes import ChangeRow

    change = ChangeRow(
        1, OPP, "O-1", "A-1", "stage", "changed", date(2026, 8, 28), date(2026, 8, 21), 1, 5,
        10, "Acme renewal: stage moved from Negotiation to Commit (CRM, 2026-08-28).",
        "S-crm", "crm/o.csv#O-1", "stage=commit", None, None, None,
    )  # fmt: skip

    def call(cid: int, day: str, fact_type: str, text: str, account: str = "acme-child"):
        return CallClaim(
            cid, "n", "signal", account, fact_type, text, "S-call", "00:01:00",
            date.fromisoformat(day), None,
        )  # fmt: skip

    budget = call(1, "2026-08-20", "other", "The champion said the budget is frozen until Q2.")
    risk = call(2, "2026-08-19", "risk", "The CFO will raise late deliveries at renewal.")
    old = call(3, "2026-07-01", "risk", "An old worry, outside the lookback window.")
    praise = call(4, "2026-08-25", "praise", "Deliveries came in right on schedule.")
    elsewhere = call(5, "2026-08-25", "risk", "Another account's risk.", account="other-co")
    assert is_negative(budget) and is_negative(risk) and not is_negative(praise)

    families = {"A-1": {"acme", "acme-child"}}
    flags = truth_flags(
        [change], [budget, risk, old, praise, elsewhere], families, window_end=date(2026, 8, 30)
    )
    assert [f.call.claim_id for f in flags] == [1]
    flags = truth_flags(
        [change], [risk, old, praise], families, window_end=date(2026, 8, 30), per_change=5
    )
    assert [f.call.claim_id for f in flags] == [2]
    backward = change.__class__(**{**change.__dict__, "direction": -1})
    assert truth_flags([backward], [budget, risk], families, window_end=date(2026, 8, 30)) == []


# --------------------------------------------------------------------------
# Against a database
# --------------------------------------------------------------------------


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


ACCOUNT = {"crm_id": "A-1", "name": "Acme Health", "kind": "system", "parent_crm_id": "",
           "domain": "acmehealth.org", "arr": "900000", "region": "Midwest",
           "status": ""}  # fmt: skip
CONTACT = {"crm_id": "C-1", "name": "Tom Ridley", "title": "Buyer", "account_crm_id": "A-1",
           "email": "tridley@acmehealth.org", "status": ""}  # fmt: skip


def _opp(stage: str, amount: str, export_date: str) -> dict[str, str]:
    return {"opportunity_id": "O-1", "account_crm_id": "A-1", "name": "Acme renewal",
            "stage": stage, "close_date": "2027-03-31", "amount": amount,
            "export_date": export_date}  # fmt: skip


@pytest.fixture()
def crm_root(tmp_path: Path) -> Path:
    root = tmp_path / "exports"
    _write(root / "crm" / "accounts.csv", [ACCOUNT])
    _write(root / "crm" / "contacts.csv", [CONTACT])
    _write(
        root / "crm" / "opportunities-2026-08-14.csv", [_opp("proposal", "100000", "2026-08-14")]
    )
    _write(root / "crm" / "opportunities-2026-08-21.csv", [_opp("commit", "125000", "2026-08-21")])
    return root


def test_each_change_is_one_claim_cited_to_both_rows_and_a_rerun_writes_nothing(
    db, crm_root: Path, tmp_path: Path
):
    connector = FileConnector(crm_root)
    Registry().import_crm(db, crm_root / "crm" / "accounts.csv", crm_root / "crm" / "contacts.csv")
    store = LocalStore(tmp_path / "store")

    report = ingest_crm_snapshots(db, connector, store)

    # Accounts and contacts in an undated file are taken in but not snapshots.
    assert (report.rows_seen, report.snapshots_recorded, report.rows_not_recorded) == (4, 2, 2)
    assert report.changes.changes_written == 2
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT ch.field, c.note_slug, c.fact_type, c.claim_text, c.source_slug, c.locator,
                   c.quote, c.valid_from, cc.source_slug, cc.locator, cc.quote
            FROM crm_changes ch JOIN claims c ON c.id = ch.claim_id
            JOIN claim_citations cc ON cc.claim_id = c.id
            ORDER BY ch.field
            """
        )
        rows = cur.fetchall()
    assert [r[0] for r in rows] == ["amount", "stage"]
    amount, stage = rows
    assert amount[3] == (
        "Acme renewal: amount rose from $100,000 to $125,000, up $25,000 or 25.0% "
        "(CRM, 2026-08-21)."
    )
    assert stage[1] == "signal-crm_change-acme-health"
    assert stage[2] == "crm_change"
    assert stage[4:8] == (
        "S-crm-opp-A1-v2",
        "crm/opportunities-2026-08-21.csv#O-1",
        "stage=commit",
        date(2026, 8, 21),
    )
    assert stage[8:] == ("S-crm-opp-A1", "crm/opportunities-2026-08-14.csv#O-1", "stage=proposal")
    with db.cursor() as cur:
        cur.execute("SELECT body_md FROM notes WHERE slug = 'signal-crm_change-acme-health'")
        (body,) = cur.fetchone()
        cur.execute("SELECT body_md FROM notes WHERE slug = 'acme-health'")
        (account_body,) = cur.fetchone()
    assert "stage moved from Proposal to Commit" in body
    assert "^[S-crm-opp-A1-v2 @ crm/opportunities-2026-08-21.csv#O-1]" in body
    assert "[[signal-crm_change-acme-health]]" in account_body

    again = ingest_crm_snapshots(db, connector, store)
    assert again.changes.changes_written == 0
    assert again.changes.already_written == 2
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM claims WHERE fact_type = 'crm_change'")
        assert cur.fetchone()[0] == 2


def test_an_export_dated_between_two_others_does_not_repeat_a_change(
    db, crm_root: Path, tmp_path: Path
):
    connector = FileConnector(crm_root)
    Registry().import_crm(db, crm_root / "crm" / "accounts.csv", crm_root / "crm" / "contacts.csv")
    store = LocalStore(tmp_path / "store")
    ingest_crm_snapshots(db, connector, store)

    late = tmp_path / "late"
    _write(
        late / "crm" / "opportunities-2026-08-18.csv", [_opp("proposal", "100000", "2026-08-18")]
    )
    report = ingest_crm_snapshots(db, FileConnector(late), store)

    assert report.changes.changes_written == 0
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM crm_changes")
        assert cur.fetchone()[0] == 2
