"""HubSpot connector (build-contracts.md "Connectors": HubSpot).

Docs relied on (fetched 2026-09-12):
- CRM v3 objects list endpoint `GET /crm/v3/objects/{objectType}`
  (`companies`, `contacts`, `deals`, `emails`), `properties` and
  `associations` query params, cursor pagination via
  `paging.next.after`, private-app bearer-token auth
  (`Authorization: Bearer <token>`):
  https://developers.hubspot.com/docs/api/crm/companies
- The `emails` engagement object's standard properties
  (`hs_timestamp`, `hs_email_subject`, `hs_email_text`,
  `hs_email_direction`, `hs_email_from_email`, `hs_email_to_email`) and
  fetching associations to companies via the `associations` query param:
  https://developers.hubspot.com/docs/api/crm/email

Not independently verified against a live portal: the exact default and
maximum page size for `GET /crm/v3/objects/*` (100 is assumed here,
HubSpot's long-documented convention for this family of endpoints; the
fetched pages didn't show the numeric limit) and whether
`hs_email_to_email` is always a single address or a comma/semicolon
delimited list in every account (treated as delimited here, matching
HubSpot's own timeline UI).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import httpx

from portolan.connectors.base import (
    SourceRef,
    Transport,
    httpx_transport,
    parse_datetime,
    request_with_retry,
    source_day,
)
from portolan.extract.schema import SourceType
from portolan.ingest.models import Participant, RawSource
from portolan.util import clock

__all__ = ["HubSpotConfig", "HubSpotConnector"]

_COMPANY_PROPERTIES = ["name", "domain", "annualrevenue", "hubspot_owner_id"]
_CONTACT_PROPERTIES = ["email", "firstname", "lastname", "jobtitle"]
_DEAL_PROPERTIES = ["dealname", "dealstage", "closedate", "amount"]
_EMAIL_PROPERTIES = [
    "hs_email_subject",
    "hs_email_text",
    "hs_email_direction",
    "hs_email_from_email",
    "hs_email_to_email",
    "hs_timestamp",
]

# HubSpot's object type names are plural in the API path ("companies",
# "emails") but the `prepare_crm` locator format wants the singular
# object name (`<object>:<crm_id>`, `portolan/ingest/prepare.py`).
_SINGULAR = {"companies": "company", "contacts": "contact", "deals": "deal", "emails": "email"}

# (object type, properties, source_id prefix, associations, RawSource.source_type)
_OBJECTS: tuple[tuple[str, list[str], str, list[str] | None, SourceType], ...] = (
    ("companies", _COMPANY_PROPERTIES, "hs-company", None, SourceType.CRM),
    ("contacts", _CONTACT_PROPERTIES, "hs-contact", ["companies"], SourceType.CRM),
    ("deals", _DEAL_PROPERTIES, "hs-deal", ["companies"], SourceType.CRM),
    ("emails", _EMAIL_PROPERTIES, "hs-email", ["companies"], SourceType.EMAIL),
)


@dataclass(frozen=True)
class HubSpotConfig:
    """A HubSpot private-app access token, never from code."""

    access_token: str
    base_url: str = "https://api.hubapi.com"
    page_size: int = 100


def _first_associated_company(record: dict) -> str | None:
    companies = (record.get("associations") or {}).get("companies") or {}
    results = companies.get("results") or []
    if results:
        return str(results[0].get("id"))
    return None


def _split_addresses(value: str) -> list[str]:
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def _crm_object_to_raw_source(source_id: str, object_type: str, record: dict) -> RawSource:
    hs_id = record["id"]
    properties = record.get("properties") or {}
    name = properties.get("name") or properties.get("dealname") or properties.get("email") or hs_id
    lines = [f"{object_type}: {name}"]
    for key, value in properties.items():
        if value is None:
            continue
        lines.append(f"{key}: {value}")

    account_hints: list[str] = []
    if object_type == "companies":
        account_hints = [hs_id]
    else:
        company_id = _first_associated_company(record)
        if company_id:
            account_hints = [company_id]

    metadata = {"object": _SINGULAR[object_type], "crm_id": hs_id}
    # The account owner, under the same key the Salesforce connector uses
    # (01-spec/actions-and-owners-2026-09-19.md).
    if object_type == "companies" and properties.get("hubspot_owner_id"):
        metadata["owner_crm_id"] = properties["hubspot_owner_id"]

    updated_raw = record.get("updatedAt")
    updated = parse_datetime(updated_raw) if updated_raw else datetime.now(UTC)
    return RawSource(
        source_id=source_id,
        source_type=SourceType.CRM,
        source_date=source_day(updated),
        participants=[],
        account_hints=account_hints,
        text="\n".join(lines),
        original_bytes=json.dumps(record, sort_keys=True).encode("utf-8"),
        original_content_type="application/json",
        metadata=metadata,
    )


def _email_object_to_raw_source(source_id: str, record: dict) -> RawSource:
    hs_id = record["id"]
    properties = record.get("properties") or {}
    participants: list[Participant] = []
    from_email = properties.get("hs_email_from_email")
    if from_email:
        participants.append(Participant(name=from_email, email=from_email, role="from"))
    for address in _split_addresses(properties.get("hs_email_to_email") or ""):
        participants.append(Participant(name=address, email=address, role="to"))

    subject = properties.get("hs_email_subject") or ""
    body = properties.get("hs_email_text") or ""
    text = f"Subject: {subject}\n\n{body}" if subject else body

    timestamp_raw = properties.get("hs_timestamp") or record.get("updatedAt")
    source_date = _parse_hubspot_timestamp(timestamp_raw)
    company_id = _first_associated_company(record)

    return RawSource(
        source_id=source_id,
        source_type=SourceType.EMAIL,
        source_date=source_date,
        participants=participants,
        account_hints=[company_id] if company_id else [],
        text=text,
        original_bytes=json.dumps(record, sort_keys=True).encode("utf-8"),
        original_content_type="application/json",
        metadata={"message_id": hs_id, "object": "email", "crm_id": hs_id},
    )


def _parse_hubspot_timestamp(value: str | int | None) -> date:
    if value is None:
        return clock.today()
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        # hs_timestamp is milliseconds-since-epoch (UTC) when set programmatically.
        return source_day(datetime.fromtimestamp(int(value) / 1000, tz=UTC))
    return source_day(parse_datetime(value))


class HubSpotConnector:
    """`build-contracts.md`'s `Connector` protocol over HubSpot's CRM v3
    objects API.

    `since=None` (a full backfill) pages the plain list endpoint, which
    has no server-side date filter. `since=<date>` (an incremental
    re-crawl) uses the CRM Search API, filtering `hs_lastmodifieddate >=
    since` server-side, so a re-crawl does not download everything. Search
    results lack the `associations` projection, so a record that needs its
    company id gets one extra GET (`_with_associations`); not confirmed
    against a live portal's search response.
    """

    name = "hubspot"

    def __init__(
        self,
        config: HubSpotConfig,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or httpx_transport(httpx.Client(timeout=30.0))
        self._sleep = sleep

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._config.access_token}"}

    def _list_objects(
        self, object_type: str, properties: list[str], associations: list[str] | None
    ) -> list[dict]:
        url = f"{self._config.base_url}/crm/v3/objects/{object_type}"
        params: dict[str, object] = {
            "limit": self._config.page_size,
            "properties": ",".join(properties),
        }
        if associations:
            params["associations"] = ",".join(associations)

        results: list[dict] = []
        while True:
            response = request_with_retry(
                self._transport,
                "GET",
                url,
                params=params,
                headers=self._headers(),
                sleep=self._sleep,
                vendor="hubspot",
            )
            payload = response.json()
            results.extend(payload.get("results", []))
            next_after = (payload.get("paging") or {}).get("next", {}).get("after")
            if not next_after:
                break
            params = dict(params, after=next_after)
        return results

    def _with_associations(self, object_type: str, record: dict, associations: list[str]) -> dict:
        """One extra GET to attach `associations` to a Search-API result
        (see class docstring) -- the list endpoint returns this
        projection directly; the Search API does not."""
        record_id = record.get("id")
        if not record_id:
            return record
        response = request_with_retry(
            self._transport,
            "GET",
            f"{self._config.base_url}/crm/v3/objects/{object_type}/{record_id}",
            params={"associations": ",".join(associations)},
            headers=self._headers(),
            sleep=self._sleep,
            vendor="hubspot",
        )
        detail = response.json()
        merged = dict(record)
        if "associations" in detail:
            merged["associations"] = detail["associations"]
        return merged

    def _search_objects(
        self,
        object_type: str,
        properties: list[str],
        associations: list[str] | None,
        since: date,
    ) -> list[dict]:
        """Server-side incremental listing via HubSpot's CRM Search API
        (class docstring). `hs_lastmodifieddate` is HubSpot's standard
        last-modified property on every CRM object type this connector reads."""
        url = f"{self._config.base_url}/crm/v3/objects/{object_type}/search"
        since_ms = int(datetime.combine(since, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
        body: dict[str, object] = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "hs_lastmodifieddate",
                            "operator": "GTE",
                            "value": since_ms,
                        }
                    ]
                }
            ],
            "properties": properties,
            "limit": self._config.page_size,
            "sorts": [{"propertyName": "hs_lastmodifieddate", "direction": "ASCENDING"}],
        }

        results: list[dict] = []
        while True:
            response = request_with_retry(
                self._transport,
                "POST",
                url,
                json=body,
                headers=self._headers(),
                sleep=self._sleep,
                vendor="hubspot",
            )
            payload = response.json()
            page_results = payload.get("results", [])
            if associations:
                page_results = [
                    self._with_associations(object_type, record, associations)
                    for record in page_results
                ]
            results.extend(page_results)
            next_after = (payload.get("paging") or {}).get("next", {}).get("after")
            if not next_after:
                break
            body = dict(body, after=next_after)
        return results

    def list_sources(self, since: date | None) -> Iterable[SourceRef]:
        for object_type, properties, prefix, associations, source_type in _OBJECTS:
            records = (
                self._list_objects(object_type, properties, associations)
                if since is None
                else self._search_objects(object_type, properties, associations, since)
            )
            for record in records:
                updated_raw = record.get("updatedAt")
                updated = parse_datetime(updated_raw) if updated_raw else datetime.now(UTC)
                if since is not None and source_day(updated) < since:
                    continue
                yield SourceRef(
                    source_id=f"{prefix}-{record['id']}",
                    source_type=source_type,
                    updated_at=updated,
                    ref={"object": object_type, "record": record},
                )

    def fetch(self, ref: SourceRef) -> RawSource:
        object_type = ref.ref["object"]
        record = ref.ref["record"]
        if object_type == "emails":
            return _email_object_to_raw_source(ref.source_id, record)
        return _crm_object_to_raw_source(ref.source_id, object_type, record)
