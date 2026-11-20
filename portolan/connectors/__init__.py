"""Real-source connectors (build-contracts.md "Connectors"): Salesforce,
HubSpot, Gong, Zoom, Clari, RSS, and the ticket connectors (Zendesk plus a
generic CSV/JSON export reader).

`FileConnector` (reads `corpus/out/`-shaped directories) is not re-exported
here or registered in `CONNECTOR_REGISTRY`.
"""

from __future__ import annotations

from portolan.connectors.base import Connector, ConnectorError, SourceRef
from portolan.connectors.registry import CONNECTOR_REGISTRY, get_connector_class

__all__ = [
    "CONNECTOR_REGISTRY",
    "Connector",
    "ConnectorError",
    "SourceRef",
    "get_connector_class",
]
