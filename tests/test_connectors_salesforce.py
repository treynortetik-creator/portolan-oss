"""Tests for `portolan.connectors.salesforce.SalesforceConnector`.

Fixtures under `tests/fixtures/connectors/salesforce/` are hand-written
JSON in the shape of Salesforce's OAuth token and SOQL REST query
responses (module docstring in `salesforce.py` cites the doc sources).
Names are Halvard Medical's synthetic customers from
`corpus/truth-table.yaml` (Ashgrove Health, Larkspur Regional Health,
Maria Chen).
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from connector_helpers import FakeTransport, json_response, load_json_fixture, sleep_recorder

from portolan.connectors.base import ConnectorError, SourceRef
from portolan.connectors.salesforce import SalesforceConfig, SalesforceConnector
from portolan.extract.schema import SourceType

QUERY_URL = "https://halvardmedical.my.salesforce.com/services/data/v61.0/query"
NEXT_RECORDS_URL = (
    "https://halvardmedical.my.salesforce.com/services/data/v61.0/query/01g1x0000000001AAA-2000"
)
TOKEN_URL = "https://login.salesforce.com/services/oauth2/token"


def _config() -> SalesforceConfig:
    return SalesforceConfig(client_id="mock-client-id", client_secret="mock-client-secret-value")


def _connector_with_full_pages() -> tuple[SalesforceConnector, FakeTransport]:
    """A connector wired to list one Account (paginated: 2 pages), one
    Contact, one Opportunity, and one EmailMessage, in that order — the
    order `SalesforceConnector.list_sources` queries objects."""
    transport = FakeTransport()
    transport.queue(TOKEN_URL, json_response(load_json_fixture("salesforce", "oauth_token.json")))
    transport.queue(
        QUERY_URL,
        json_response(load_json_fixture("salesforce", "accounts_page1.json")),
        json_response(load_json_fixture("salesforce", "contacts.json")),
        json_response(load_json_fixture("salesforce", "opportunities.json")),
        json_response(load_json_fixture("salesforce", "email_messages.json")),
    )
    transport.queue(
        NEXT_RECORDS_URL, json_response(load_json_fixture("salesforce", "accounts_page2.json"))
    )
    sleep, _ = sleep_recorder()
    connector = SalesforceConnector(_config(), transport=transport, sleep=sleep)
    return connector, transport


def test_list_sources_authenticates_and_yields_every_object_type():
    connector, transport = _connector_with_full_pages()

    refs = list(connector.list_sources(since=None))

    by_type = {}
    for ref in refs:
        by_type.setdefault(ref.source_type, []).append(ref)

    assert len(by_type[SourceType.CRM]) == 4  # 2 accounts + 1 contact + 1 opportunity
    assert len(by_type[SourceType.EMAIL]) == 1
    assert {r.source_id for r in by_type[SourceType.CRM]} == {
        "sf-account-0011x000003Ashgv",
        "sf-account-0011x000003Lrksp",
        "sf-contact-0031x000004MChen",
        "sf-opportunity-0061x000002OppAG",
    }
    assert by_type[SourceType.EMAIL][0].source_id == "sf-email-02s1x000000EmlAB"

    token_calls = [c for c in transport.calls if c["url"] == TOKEN_URL]
    assert len(token_calls) == 1  # authenticate once, cache the token


def test_pagination_across_two_pages_follows_next_records_url():
    connector, transport = _connector_with_full_pages()

    refs = list(connector.list_sources(since=None))
    account_ids = {r.source_id for r in refs if r.source_id.startswith("sf-account-")}

    assert account_ids == {"sf-account-0011x000003Ashgv", "sf-account-0011x000003Lrksp"}
    next_records_calls = [c for c in transport.calls if c["url"] == NEXT_RECORDS_URL]
    assert len(next_records_calls) == 1


def test_since_filters_soql_on_last_modified_date():
    connector, transport = _connector_with_full_pages()

    list(connector.list_sources(since=date(2026, 8, 1)))

    query_calls = [c for c in transport.calls if c["url"] == QUERY_URL]
    account_query = query_calls[0]["params"]["q"]
    assert "WHERE LastModifiedDate >= 2026-08-01T00:00:00Z" in account_query


def test_fetch_account_row_is_crm_type_with_locator_metadata():
    connector, _ = _connector_with_full_pages()
    refs = list(connector.list_sources(since=None))
    account_ref = next(r for r in refs if r.source_id == "sf-account-0011x000003Ashgv")

    raw = connector.fetch(account_ref)

    assert raw.source_type == SourceType.CRM
    assert raw.metadata == {"object": "account", "crm_id": "0011x000003Ashgv"}
    assert raw.account_hints == ["0011x000003Ashgv"]
    assert "Ashgrove Health" in raw.text
    assert raw.original_bytes is not None


def test_fetch_account_row_maps_owner_id_into_metadata_when_present():
    """Owners, Phase 1 (01-spec/actions-and-owners-2026-09-19.md build
    item 4): Larkspur Regional Health's fixture row (accounts_page2.json)
    carries `OwnerId` -- it must land on `raw.metadata['owner_crm_id']`,
    read-only, nothing else changed."""
    connector, _ = _connector_with_full_pages()
    refs = list(connector.list_sources(since=None))
    larkspur_ref = next(r for r in refs if r.source_id == "sf-account-0011x000003Lrksp")

    raw = connector.fetch(larkspur_ref)

    assert raw.metadata == {
        "object": "account",
        "crm_id": "0011x000003Lrksp",
        "owner_crm_id": "0051x000000SamOk",
    }


def test_fetch_contact_row_carries_account_hint_from_account_id():
    connector, _ = _connector_with_full_pages()
    refs = list(connector.list_sources(since=None))
    contact_ref = next(r for r in refs if r.source_id == "sf-contact-0031x000004MChen")

    raw = connector.fetch(contact_ref)

    assert raw.account_hints == ["0011x000003Ashgv"]
    assert raw.metadata == {"object": "contact", "crm_id": "0031x000004MChen"}


def test_fetch_email_message_yields_email_source_with_participants():
    connector, _ = _connector_with_full_pages()
    refs = list(connector.list_sources(since=None))
    email_ref = next(r for r in refs if r.source_type == SourceType.EMAIL)

    raw = connector.fetch(email_ref)

    assert raw.source_type == SourceType.EMAIL
    assert raw.source_date == date(2026, 8, 7)
    emails = {p.email for p in raw.participants}
    assert emails == {"abell@cobaltridgemed.org", "s.okafor@halvardmedical.com"}
    assert raw.metadata["message_id"] == "02s1x000000EmlAB"
    assert "three days late" in raw.text
    assert raw.account_hints == ["0011x000003CoblRg"]


def test_auth_error_surfaces_as_connector_error_without_the_secret():
    transport = FakeTransport()
    transport.queue(
        TOKEN_URL,
        json_response(
            {"error": "invalid_client", "error_description": "invalid client credentials"},
            status_code=401,
        ),
    )
    sleep, _ = sleep_recorder()
    config = _config()
    connector = SalesforceConnector(config, transport=transport, sleep=sleep)

    with pytest.raises(ConnectorError) as excinfo:
        list(connector.list_sources(since=None))

    message = str(excinfo.value)
    assert "invalid client credentials" in message
    assert config.client_secret not in message


def test_a_429_then_success_retries_and_returns_data():
    transport = FakeTransport()
    transport.queue(TOKEN_URL, json_response(load_json_fixture("salesforce", "oauth_token.json")))
    transport.queue(
        QUERY_URL,
        json_response({}, status_code=429, headers={"Retry-After": "0"}),
        json_response(load_json_fixture("salesforce", "accounts_page2.json")),
        json_response(load_json_fixture("salesforce", "contacts.json")),
        json_response(load_json_fixture("salesforce", "opportunities.json")),
        json_response(load_json_fixture("salesforce", "email_messages.json")),
    )
    sleep, waits = sleep_recorder()
    connector = SalesforceConnector(_config(), transport=transport, sleep=sleep)

    refs = list(connector.list_sources(since=None))

    assert any(r.source_id == "sf-account-0011x000003Lrksp" for r in refs)
    assert waits == [0.0]


def test_next_records_url_refuses_a_foreign_host():
    """Review finding F8: a compromised or spoofed SOQL response could
    point `nextRecordsUrl` at an attacker's host and this connector would
    still attach its bearer token. `require_allowed_origin` must refuse
    before the transport is ever called with that URL."""
    transport = FakeTransport()
    transport.queue(TOKEN_URL, json_response(load_json_fixture("salesforce", "oauth_token.json")))
    transport.queue(
        QUERY_URL,
        json_response(
            {
                "totalSize": 1,
                "done": False,
                "nextRecordsUrl": "https://evil.example.com/steal-the-bearer-token",
                "records": [],
            }
        ),
    )
    sleep, _ = sleep_recorder()
    connector = SalesforceConnector(_config(), transport=transport, sleep=sleep)

    with pytest.raises(ConnectorError, match=r"evil\.example\.com"):
        list(connector.list_sources(since=None))

    assert not any(c["url"].startswith("https://evil.example.com") for c in transport.calls)


def test_fetch_uses_ref_object_type_to_pick_the_conversion():
    connector = SalesforceConnector(_config(), transport=FakeTransport())
    ref = SourceRef(
        source_id="sf-account-x",
        source_type=SourceType.CRM,
        updated_at=datetime(2026, 8, 20, 14, 3),
        ref={
            "object": "Account",
            "record": {
                "Id": "0011x000003Ashgv",
                "Name": "Ashgrove Health",
                "LastModifiedDate": "2026-08-20T14:03:00.000+0000",
            },
        },
    )

    raw = connector.fetch(ref)

    assert raw.metadata["object"] == "account"
