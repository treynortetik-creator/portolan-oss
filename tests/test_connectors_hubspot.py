"""Tests for `portolan.connectors.hubspot.HubSpotConnector`.

Fixtures under `tests/fixtures/connectors/hubspot/` mirror HubSpot's CRM
v3 objects list response shape (module docstring in `hubspot.py` cites
the doc sources). Names are Halvard Medical's synthetic customers from
`corpus/truth-table.yaml`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from connector_helpers import FakeTransport, json_response, load_json_fixture

from portolan.connectors.base import ConnectorError
from portolan.connectors.hubspot import HubSpotConfig, HubSpotConnector
from portolan.extract.schema import SourceType

BASE = "https://api.hubapi.com"


def _config() -> HubSpotConfig:
    return HubSpotConfig(access_token="mock-private-app-token")


def _connector() -> tuple[HubSpotConnector, FakeTransport]:
    transport = FakeTransport()
    transport.queue(
        f"{BASE}/crm/v3/objects/companies",
        json_response(load_json_fixture("hubspot", "companies_page1.json")),
        json_response(load_json_fixture("hubspot", "companies_page2.json")),
    )
    transport.queue(
        f"{BASE}/crm/v3/objects/contacts",
        json_response(load_json_fixture("hubspot", "contacts.json")),
    )
    transport.queue(
        f"{BASE}/crm/v3/objects/deals", json_response(load_json_fixture("hubspot", "deals.json"))
    )
    transport.queue(
        f"{BASE}/crm/v3/objects/emails", json_response(load_json_fixture("hubspot", "emails.json"))
    )
    return HubSpotConnector(_config(), transport=transport), transport


def test_list_sources_yields_companies_contacts_deals_and_emails():
    connector, _ = _connector()

    refs = list(connector.list_sources(since=None))

    ids = {r.source_id for r in refs}
    assert ids == {
        "hs-company-5001",
        "hs-company-5002",
        "hs-contact-6001",
        "hs-deal-7001",
        "hs-email-8001",
    }
    email_ref = next(r for r in refs if r.source_id == "hs-email-8001")
    assert email_ref.source_type == SourceType.EMAIL


def test_pagination_across_two_pages_follows_paging_next_after():
    connector, transport = _connector()

    refs = list(connector.list_sources(since=None))

    company_ids = {r.source_id for r in refs if r.source_id.startswith("hs-company-")}
    assert company_ids == {"hs-company-5001", "hs-company-5002"}
    calls = [c for c in transport.calls if c["url"] == f"{BASE}/crm/v3/objects/companies"]
    assert len(calls) == 2
    assert calls[1]["params"]["after"] == "cursor-page-2"


def test_since_uses_the_search_api_with_a_server_side_date_filter():
    """Review finding F17: the plain list endpoint has no server-side
    date filter, so an incremental (`since=<date>`) crawl used to page
    every object in full and filter client-side after downloading it
    all. `since=<date>` now goes through the CRM Search API instead,
    with `hs_lastmodifieddate >= since` as a server-side filter -- the
    plain list endpoint is never touched for that case at all.
    """

    def search_url(object_type: str) -> str:
        return f"{BASE}/crm/v3/objects/{object_type}/search"

    transport = FakeTransport()
    transport.queue(
        search_url("companies"),
        json_response(load_json_fixture("hubspot", "companies_page1.json")),
        json_response(load_json_fixture("hubspot", "companies_page2.json")),
    )
    transport.queue(
        search_url("contacts"), json_response(load_json_fixture("hubspot", "contacts.json"))
    )
    transport.queue(search_url("deals"), json_response(load_json_fixture("hubspot", "deals.json")))
    transport.queue(
        search_url("emails"), json_response(load_json_fixture("hubspot", "emails.json"))
    )
    # contacts/deals/emails associate to "companies" -- the Search API
    # doesn't return that projection, so one follow-up GET per result
    # attaches it (HubSpotConnector._with_associations).
    transport.queue(
        f"{BASE}/crm/v3/objects/contacts/6001",
        json_response({"id": "6001", "associations": {"companies": {"results": [{"id": "5001"}]}}}),
    )
    transport.queue(
        f"{BASE}/crm/v3/objects/deals/7001",
        json_response({"id": "7001", "associations": {"companies": {"results": [{"id": "5001"}]}}}),
    )
    transport.queue(
        f"{BASE}/crm/v3/objects/emails/8001",
        json_response({"id": "8001", "associations": {"companies": {"results": [{"id": "5003"}]}}}),
    )
    connector = HubSpotConnector(_config(), transport=transport)
    since = date(2026, 8, 1)

    refs = list(connector.list_sources(since=since))

    ids = {r.source_id for r in refs}
    assert ids == {
        "hs-company-5001",
        "hs-company-5002",
        "hs-contact-6001",
        "hs-deal-7001",
        "hs-email-8001",
    }

    search_call = next(c for c in transport.calls if c["url"] == search_url("companies"))
    date_filter = search_call["json"]["filterGroups"][0]["filters"][0]
    assert date_filter["propertyName"] == "hs_lastmodifieddate"
    assert date_filter["operator"] == "GTE"
    assert date_filter["value"] == int(datetime(2026, 8, 1, tzinfo=UTC).timestamp() * 1000)

    assert not any(c["url"] == f"{BASE}/crm/v3/objects/companies" for c in transport.calls)


def test_fetch_company_row_is_crm_type_with_locator_metadata():
    connector, _ = _connector()
    refs = list(connector.list_sources(since=None))
    ref = next(r for r in refs if r.source_id == "hs-company-5001")

    raw = connector.fetch(ref)

    assert raw.source_type == SourceType.CRM
    assert raw.metadata == {"object": "company", "crm_id": "5001"}
    assert raw.account_hints == ["5001"]
    assert "Ashgrove Health" in raw.text


def test_fetch_company_row_maps_hubspot_owner_id_into_metadata_when_present():
    """Owners, Phase 1 (01-spec/actions-and-owners-2026-09-19.md build
    item 4): Larkspur Regional Health's fixture row (companies_page2.json)
    carries `hubspot_owner_id` -- it must land on
    `raw.metadata['owner_crm_id']`, read-only, nothing else changed."""
    connector, _ = _connector()
    refs = list(connector.list_sources(since=None))
    ref = next(r for r in refs if r.source_id == "hs-company-5002")

    raw = connector.fetch(ref)

    assert raw.metadata == {"object": "company", "crm_id": "5002", "owner_crm_id": "700001"}


def test_fetch_contact_row_carries_associated_company_as_account_hint():
    connector, _ = _connector()
    refs = list(connector.list_sources(since=None))
    ref = next(r for r in refs if r.source_id == "hs-contact-6001")

    raw = connector.fetch(ref)

    assert raw.account_hints == ["5001"]
    assert raw.metadata == {"object": "contact", "crm_id": "6001"}


def test_fetch_email_object_yields_email_source_with_participants():
    connector, _ = _connector()
    refs = list(connector.list_sources(since=None))
    ref = next(r for r in refs if r.source_type == SourceType.EMAIL)

    raw = connector.fetch(ref)

    assert raw.source_type == SourceType.EMAIL
    assert raw.source_date == date(2026, 8, 7)
    emails = {p.email for p in raw.participants}
    assert emails == {"abell@cobaltridgemed.org", "s.okafor@halvardmedical.com"}
    assert raw.account_hints == ["5003"]
    assert "three days late" in raw.text


def test_auth_error_surfaces_as_connector_error_without_the_token():
    transport = FakeTransport()
    transport.queue(
        f"{BASE}/crm/v3/objects/companies",
        json_response(load_json_fixture("hubspot", "auth_error.json"), status_code=401),
    )
    config = _config()
    connector = HubSpotConnector(config, transport=transport)

    with pytest.raises(ConnectorError) as excinfo:
        list(connector.list_sources(since=None))

    message = str(excinfo.value)
    assert "required scopes" in message
    assert config.access_token not in message
