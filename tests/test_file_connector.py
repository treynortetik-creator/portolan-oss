"""Tests for `portolan.connectors.file_connector.FileConnector` against the
real `corpus/out` layout (no network, no fakes needed -- this connector
only ever reads local files).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from portolan.connectors.base import Connector, SourceRef
from portolan.connectors.file_connector import CrmPaths, FileConnector
from portolan.extract.schema import SourceType

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_OUT = REPO_ROOT / "corpus" / "out"


@pytest.fixture(scope="module")
def connector() -> FileConnector:
    return FileConnector(CORPUS_OUT)


@pytest.fixture(scope="module")
def refs_by_id(connector: FileConnector) -> dict[str, SourceRef]:
    return {ref.source_id: ref for ref in connector.list_sources()}


def test_file_connector_satisfies_the_connector_protocol(connector: FileConnector) -> None:
    assert isinstance(connector, Connector)
    assert connector.name == "file"


# --------------------------------------------------------------------
# list_sources: counts and ids
# --------------------------------------------------------------------


def test_lists_one_ref_per_call_file(refs_by_id: dict[str, SourceRef]) -> None:
    call_files = sorted((CORPUS_OUT / "calls").glob("*.txt"))
    call_ids = {ref.source_id for ref in refs_by_id.values() if ref.source_type == SourceType.CALL}
    assert call_ids == {p.stem for p in call_files}


def test_lists_one_ref_per_ticket_file(refs_by_id: dict[str, SourceRef]) -> None:
    ticket_files = sorted((CORPUS_OUT / "tickets").glob("*.json"))
    ticket_ids = {
        ref.source_id for ref in refs_by_id.values() if ref.source_type == SourceType.TICKET
    }
    assert ticket_ids == {p.stem for p in ticket_files}


def test_lists_one_ref_per_email_file(refs_by_id: dict[str, SourceRef]) -> None:
    email_files = sorted((CORPUS_OUT / "email").glob("*.txt"))
    email_ids = {
        ref.source_id for ref in refs_by_id.values() if ref.source_type == SourceType.EMAIL
    }
    assert email_ids == {p.stem for p in email_files}
    assert "S-email-040" in email_ids
    assert "S-email-061" in email_ids
    assert "S-email-021-thread" in email_ids


def test_lists_one_ref_per_rss_item_not_per_file(refs_by_id: dict[str, SourceRef]) -> None:
    expected_ids: set[str] = set()
    for feed_path in (CORPUS_OUT / "rss").glob("*.json"):
        items = json.loads(feed_path.read_text(encoding="utf-8"))
        expected_ids.update(item["id"] for item in items)
    rss_ids = {ref.source_id for ref in refs_by_id.values() if ref.source_type == SourceType.RSS}
    assert rss_ids == expected_ids
    assert len(expected_ids) > 2  # confirms multiple items come from one feed file
    assert "S-rss-004" in rss_ids


def test_lists_one_ref_per_crm_row_including_both_opportunity_snapshots(
    refs_by_id: dict[str, SourceRef],
) -> None:
    import csv

    crm_refs = [ref for ref in refs_by_id.values() if ref.source_type == SourceType.CRM]

    def _row_count(name: str) -> int:
        with (CORPUS_OUT / "crm" / name).open(encoding="utf-8") as fh:
            return sum(1 for _ in csv.DictReader(fh))

    expected = (
        _row_count("accounts.csv")
        + _row_count("contacts.csv")
        + _row_count("opportunities.csv")
        + _row_count("opportunities-2026-08-21.csv")
    )
    assert len(crm_refs) == expected  # one ref per real CRM CSV row, both opportunity snapshots

    assert "S-crm-A1002" in refs_by_id
    assert "S-crm-A1010" in refs_by_id
    assert "S-crm-opp-A1010" in refs_by_id
    assert "S-crm-opp-A1010-v2" in refs_by_id
    # The two opportunity snapshots for the same account are distinct
    # sources, dated by their own export_date, not deduped into one.
    first = refs_by_id["S-crm-opp-A1010"]
    second = refs_by_id["S-crm-opp-A1010-v2"]
    assert first.updated_at.date() == date(2026, 4, 15)
    assert second.updated_at.date() == date(2026, 8, 21)


def test_crm_paths_finds_accounts_and_contacts(connector: FileConnector) -> None:
    paths = connector.crm_paths()
    assert isinstance(paths, CrmPaths)
    assert paths.accounts == CORPUS_OUT / "crm" / "accounts.csv"
    assert paths.contacts == CORPUS_OUT / "crm" / "contacts.csv"


def test_crm_paths_is_none_when_directory_has_no_crm_folder(tmp_path: Path) -> None:
    empty_connector = FileConnector(tmp_path)
    assert empty_connector.crm_paths() is None


# --------------------------------------------------------------------
# fetch: participants, dates, account_hints, bytes
# --------------------------------------------------------------------


def test_fetch_call_parses_speaker_names_as_participants(
    connector: FileConnector, refs_by_id: dict[str, SourceRef]
) -> None:
    raw = connector.fetch(refs_by_id["S-call-009"])
    names = {p.name for p in raw.participants}
    assert "Felix Arden" in names
    assert "Sam Okafor" in names
    assert raw.source_type == SourceType.CALL
    assert raw.source_date == date(2026, 8, 5)
    assert raw.account_hints == ["larkspur-memorial"]
    assert raw.original_bytes == (CORPUS_OUT / "calls" / "S-call-009.txt").read_bytes()


def test_fetch_ticket_uses_requester_as_the_only_participant(
    connector: FileConnector, refs_by_id: dict[str, SourceRef]
) -> None:
    raw = connector.fetch(refs_by_id["S-ticket-019"])
    assert raw.source_type == SourceType.TICKET
    assert [p.name for p in raw.participants] == ["Owen Castellane"]
    assert raw.source_date == date(2026, 5, 12)
    assert raw.metadata["comments"][0]["number"] == 1
    assert raw.account_hints == ["st-aldric"]
    assert raw.original_bytes == (CORPUS_OUT / "tickets" / "S-ticket-019.json").read_bytes()


def test_fetch_email_parses_from_and_to_participants(
    connector: FileConnector, refs_by_id: dict[str, SourceRef]
) -> None:
    raw = connector.fetch(refs_by_id["S-email-021"])
    assert raw.source_type == SourceType.EMAIL
    emails = {p.email for p in raw.participants if p.email}
    assert "ocastellane@staldric.org" in emails
    assert "p.natarajan@halvardmedical.com" in emails
    assert raw.source_date == date(2026, 4, 22)
    assert raw.metadata["message_id"] == "S-email-021@halvardmedical.com"
    assert raw.original_bytes == (CORPUS_OUT / "email" / "S-email-021.txt").read_bytes()


def test_fetch_rss_has_no_participants_and_per_item_bytes(
    connector: FileConnector, refs_by_id: dict[str, SourceRef]
) -> None:
    raw = connector.fetch(refs_by_id["S-rss-004"])
    assert raw.source_type == SourceType.RSS
    assert raw.participants == []
    assert raw.source_date == date(2026, 8, 26)
    assert "Corvane" in raw.text
    # Per-item bytes, not the whole feed file -- see FileConnector's module
    # docstring: two items sharing one file must not share a content hash.
    feed_bytes = (CORPUS_OUT / "rss" / "industry-news.json").read_bytes()
    assert raw.original_bytes != feed_bytes


def test_fetch_crm_has_no_participants_and_per_row_bytes(
    connector: FileConnector, refs_by_id: dict[str, SourceRef]
) -> None:
    raw = connector.fetch(refs_by_id["S-crm-A1010"])
    assert raw.source_type == SourceType.CRM
    assert raw.participants == []
    # "path" was added (review fix wave F8) so prepare.py can build the
    # manifest-exact CRM locator `crm/accounts.csv#A-1010` instead of the
    # older `account:A-1010` form.
    assert raw.metadata == {"object": "account", "crm_id": "A-1010", "path": "crm/accounts.csv"}
    assert "Ashgrove Health" in raw.text
    csv_bytes = (CORPUS_OUT / "crm" / "accounts.csv").read_bytes()
    assert raw.original_bytes != csv_bytes


def test_two_crm_rows_from_the_same_file_get_different_content_hashes(
    connector: FileConnector, refs_by_id: dict[str, SourceRef]
) -> None:
    """Each CRM row's original is its own row, not the shared
    `accounts.csv` (module docstring): with the file's bytes, every row
    would store the whole file as its original, and an edit to any one row
    would change every row's content hash, so `portolan.ingest.intake`
    (keyed on `(source_id, content_hash)`) would re-version all of them."""
    a = connector.fetch(refs_by_id["S-crm-A1002"])
    b = connector.fetch(refs_by_id["S-crm-A1010"])
    assert a.original_bytes != b.original_bytes


def test_two_rss_items_from_the_same_file_get_different_content_hashes(
    connector: FileConnector, refs_by_id: dict[str, SourceRef]
) -> None:
    industry_items = [
        ref for sid, ref in refs_by_id.items() if sid.startswith("S-rss-") and "n0" not in sid
    ]
    fetched = [connector.fetch(ref) for ref in industry_items[:2]]
    assert fetched[0].original_bytes != fetched[1].original_bytes


def test_account_hints_come_from_manifest_when_present(
    connector: FileConnector, refs_by_id: dict[str, SourceRef]
) -> None:
    raw = connector.fetch(refs_by_id["S-call-009"])
    assert raw.account_hints == ["larkspur-memorial"]

    manifest_less = FileConnector(CORPUS_OUT.parent)  # corpus/ has no manifest.json of its own
    assert manifest_less._manifest_by_id == {}


def test_since_filters_out_older_sources(connector: FileConnector) -> None:
    all_refs = list(connector.list_sources())
    cutoff = date(2026, 8, 1)
    recent_refs = list(connector.list_sources(since=cutoff))
    assert len(recent_refs) < len(all_refs)
    assert all(ref.updated_at.date() >= cutoff for ref in recent_refs)


def test_fetch_raises_for_a_source_type_it_does_not_carry(connector: FileConnector) -> None:
    from datetime import UTC, datetime

    bogus_ref = SourceRef(
        source_id="nope", source_type=SourceType.DOC, updated_at=datetime.now(UTC), ref={}
    )
    with pytest.raises(ValueError, match="cannot fetch source_type"):
        connector.fetch(bogus_ref)
