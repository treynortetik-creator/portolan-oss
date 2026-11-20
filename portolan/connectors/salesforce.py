"""Salesforce connector (build-contracts.md "Connectors": Salesforce;
portolan-spec-v2.md sec 5.3: "the emails the CRM has logged").

Docs relied on (researched 2026-09-12):
- OAuth 2.0 client-credentials and refresh-token flows: token endpoint
  `{login_url}/services/oauth2/token`, JSON response carrying
  `access_token` and `instance_url`.
  https://help.salesforce.com/s/articleView?id=sf.remoteaccess_oauth_client_credentials_flow.htm
  (the REST developer guide itself could not be fetched; corroborated by
  independent write-ups, e.g. https://medium.com/@dcheng_93016/understanding-salesforce-oauth-2-0-flows-a-practical-guide-ab2986eca610)
- SOQL REST query endpoint `GET {instance_url}/services/data/v{version}/query`
  with a `q` parameter, JSON response `{totalSize, done, records,
  nextRecordsUrl}`, `nextRecordsUrl` only present while `done` is false
  (`.../api_rest/dome_query.htm`, corroborated the same way).
- `EmailMessage` is Salesforce's standard object for CRM-logged email
  (populated by Einstein Activity Capture or a logging integration); the
  fields this connector reads (`Subject`, `TextBody`, `FromAddress`,
  `ToAddress`, `MessageDate`, `RelatedToId`) are that object's documented
  standard fields.

Not independently verified against a live org: some orgs surface only
`HtmlBody` (not `TextBody`) on `EmailMessage`, and `ToAddress` can be
absent when a logging integration instead writes child
`EmailMessageRelation` rows — this connector reads the fields the build
contract names and should be checked against a real sandbox before
go-live.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlsplit

import httpx

from portolan.connectors.base import (
    SourceRef,
    Transport,
    httpx_transport,
    parse_datetime,
    request_with_reauth,
    request_with_retry,
    require_allowed_origin,
    source_day,
)
from portolan.extract.schema import SourceType
from portolan.ingest.models import Participant, RawSource

__all__ = ["SalesforceConfig", "SalesforceConnector"]

DEFAULT_API_VERSION = "v61.0"

_ACCOUNT_FIELDS = "Id, Name, ParentId, Website, AnnualRevenue, OwnerId, LastModifiedDate"
_CONTACT_FIELDS = "Id, Name, Email, Title, AccountId, LastModifiedDate"
_OPPORTUNITY_FIELDS = "Id, Name, CloseDate, StageName, Amount, AccountId, LastModifiedDate"
_EMAIL_FIELDS = (
    "Id, Subject, TextBody, FromAddress, ToAddress, MessageDate, RelatedToId, LastModifiedDate"
)

# (SObject, field list, source_id prefix, RawSource.source_type)
_OBJECTS: tuple[tuple[str, str, str, SourceType], ...] = (
    ("Account", _ACCOUNT_FIELDS, "sf-account", SourceType.CRM),
    ("Contact", _CONTACT_FIELDS, "sf-contact", SourceType.CRM),
    ("Opportunity", _OPPORTUNITY_FIELDS, "sf-opportunity", SourceType.CRM),
    ("EmailMessage", _EMAIL_FIELDS, "sf-email", SourceType.EMAIL),
)


@dataclass(frozen=True)
class SalesforceConfig:
    """Credentials only ever come from here — never hardcoded (build
    contract: "credentials from a `SalesforceConfig` dataclass, never
    from code"). Set `refresh_token` to use the refresh-token flow;
    leave it `None` for client-credentials."""

    client_id: str
    client_secret: str
    login_url: str = "https://login.salesforce.com"
    refresh_token: str | None = None
    api_version: str = DEFAULT_API_VERSION


def _split_addresses(value: str) -> list[str]:
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def _crm_row_to_raw_source(source_id: str, sobject: str, record: dict) -> RawSource:
    crm_id = record["Id"]
    name = record.get("Name") or crm_id
    lines = [f"{sobject}: {name}"]
    for key, value in record.items():
        if key in ("attributes", "Id", "Name") or value is None:
            continue
        lines.append(f"{key}: {value}")

    account_hints: list[str] = []
    if sobject == "Account":
        account_hints = [crm_id]
    elif record.get("AccountId"):
        account_hints = [record["AccountId"]]

    metadata = {"object": sobject.lower(), "crm_id": crm_id}
    # The account owner, under the same key the HubSpot connector uses
    # (01-spec/actions-and-owners-2026-09-19.md). Only Account has an owner
    # field in this connector's field lists.
    if sobject == "Account" and record.get("OwnerId"):
        metadata["owner_crm_id"] = record["OwnerId"]

    updated = parse_datetime(record["LastModifiedDate"])
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


def _email_message_to_raw_source(source_id: str, record: dict) -> RawSource:
    crm_id = record["Id"]
    participants: list[Participant] = []
    from_address = record.get("FromAddress")
    if from_address:
        participants.append(Participant(name=from_address, email=from_address, role="from"))
    for address in _split_addresses(record.get("ToAddress") or ""):
        participants.append(Participant(name=address, email=address, role="to"))

    subject = record.get("Subject") or ""
    body = record.get("TextBody") or ""
    text = f"Subject: {subject}\n\n{body}" if subject else body

    message_date = record.get("MessageDate") or record["LastModifiedDate"]
    source_date = source_day(parse_datetime(message_date))
    account_hints = [record["RelatedToId"]] if record.get("RelatedToId") else []

    return RawSource(
        source_id=source_id,
        source_type=SourceType.EMAIL,
        source_date=source_date,
        participants=participants,
        account_hints=account_hints,
        text=text,
        original_bytes=json.dumps(record, sort_keys=True).encode("utf-8"),
        original_content_type="application/json",
        metadata={"message_id": crm_id, "object": "emailmessage", "crm_id": crm_id},
    )


class SalesforceConnector:
    """`build-contracts.md`'s `Connector` protocol over the Salesforce
    REST API. `since` filters every object query on `LastModifiedDate`
    (build contract: "Support `since` on `LastModifiedDate`")."""

    name = "salesforce"

    def __init__(
        self,
        config: SalesforceConfig,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or httpx_transport(httpx.Client(timeout=30.0))
        self._sleep = sleep
        self._access_token: str | None = None
        self._instance_url: str | None = None

    def _authenticate(self) -> None:
        if self._access_token is not None:
            return
        cfg = self._config
        if cfg.refresh_token:
            data = {
                "grant_type": "refresh_token",
                "client_id": cfg.client_id,
                "client_secret": cfg.client_secret,
                "refresh_token": cfg.refresh_token,
            }
        else:
            data = {
                "grant_type": "client_credentials",
                "client_id": cfg.client_id,
                "client_secret": cfg.client_secret,
            }
        response = request_with_retry(
            self._transport,
            "POST",
            f"{cfg.login_url}/services/oauth2/token",
            data=data,
            sleep=self._sleep,
            vendor="salesforce",
        )
        payload = response.json()
        self._access_token = payload["access_token"]
        self._instance_url = payload["instance_url"]

    def _headers(self) -> dict[str, str]:
        self._authenticate()
        return {"Authorization": f"Bearer {self._access_token}"}

    def _force_reauth_headers(self) -> dict[str, str]:
        """Discard the cached token and authenticate again: the
        `reauthenticate` callback `request_with_reauth` calls on a 401."""
        self._access_token = None
        return self._headers()

    def _soql(self, query: str) -> list[dict]:
        self._authenticate()
        url = f"{self._instance_url}/services/data/{self._config.api_version}/query"
        response = request_with_reauth(
            self._transport,
            "GET",
            url,
            params={"q": query},
            headers=self._headers(),
            reauthenticate=self._force_reauth_headers,
            sleep=self._sleep,
            vendor="salesforce",
        )
        payload = response.json()
        records: list[dict] = list(payload.get("records", []))
        while not payload.get("done", True):
            next_url = payload["nextRecordsUrl"]
            if next_url.startswith("http"):
                # The bearer token goes with this request: only the host we
                # authenticated against.
                require_allowed_origin(
                    next_url,
                    allowed_hosts=[urlsplit(self._instance_url or "").hostname or ""],
                    vendor="salesforce",
                )
                full_url = next_url
            else:
                full_url = f"{self._instance_url}{next_url}"
            response = request_with_reauth(
                self._transport,
                "GET",
                full_url,
                headers=self._headers(),
                reauthenticate=self._force_reauth_headers,
                sleep=self._sleep,
                vendor="salesforce",
            )
            payload = response.json()
            records.extend(payload.get("records", []))
        return records

    def list_sources(self, since: date | None) -> Iterable[SourceRef]:
        cutoff = f"{since.isoformat()}T00:00:00Z" if since is not None else None
        for sobject, fields, prefix, source_type in _OBJECTS:
            query = f"SELECT {fields} FROM {sobject}"
            if cutoff is not None:
                query += f" WHERE LastModifiedDate >= {cutoff}"
            for record in self._soql(query):
                yield SourceRef(
                    source_id=f"{prefix}-{record['Id']}",
                    source_type=source_type,
                    updated_at=parse_datetime(record["LastModifiedDate"]),
                    ref={"object": sobject, "record": record},
                )

    def fetch(self, ref: SourceRef) -> RawSource:
        sobject = ref.ref["object"]
        record = ref.ref["record"]
        if sobject == "EmailMessage":
            return _email_message_to_raw_source(ref.source_id, record)
        return _crm_row_to_raw_source(ref.source_id, sobject, record)
