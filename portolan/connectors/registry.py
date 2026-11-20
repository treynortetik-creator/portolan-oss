"""Connector registry (build-contracts.md "Connectors": `registry.py`
"mapping connector names to classes"): turns a config string
(`--connector salesforce`) into a `Connector` class. `FileConnector` is not
registered here.
"""

from __future__ import annotations

from portolan.connectors.base import Connector
from portolan.connectors.clari import ClariConnector
from portolan.connectors.gong import GongConnector
from portolan.connectors.hubspot import HubSpotConnector
from portolan.connectors.rss import RSSConnector
from portolan.connectors.salesforce import SalesforceConnector
from portolan.connectors.tickets import ExportTicketConnector, ZendeskConnector
from portolan.connectors.zoom import ZoomConnector

__all__ = ["CONNECTOR_REGISTRY", "EXPERIMENTAL_CONNECTORS", "get_connector_class"]

CONNECTOR_REGISTRY: dict[str, type[Connector]] = {
    "salesforce": SalesforceConnector,
    "hubspot": HubSpotConnector,
    "gong": GongConnector,
    "zoom": ZoomConnector,
    "clari": ClariConnector,
    "rss": RSSConnector,
    "zendesk": ZendeskConnector,
    "ticket_export": ExportTicketConnector,
}

# Connectors built without the vendor's own API reference (`clari.py`'s module
# docstring), so a caller can warn or gate. Not a list of the unverified ones:
# the other connector modules also list what is unchecked against a live account.
EXPERIMENTAL_CONNECTORS: frozenset[str] = frozenset({"clari"})


def get_connector_class(name: str) -> type[Connector]:
    """Look up a connector class by its registered name, raising a
    `KeyError` that lists what is registered on an unknown name."""
    try:
        return CONNECTOR_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(CONNECTOR_REGISTRY))
        raise KeyError(f"unknown connector {name!r}; available: {available}") from exc
