"""Tests for `portolan.connectors.registry` (build-contracts.md
"Connectors": `registry.py` "mapping connector names to classes")."""

from __future__ import annotations

import pytest

from portolan.connectors.clari import ClariConnector
from portolan.connectors.gong import GongConnector
from portolan.connectors.hubspot import HubSpotConnector
from portolan.connectors.registry import CONNECTOR_REGISTRY, get_connector_class
from portolan.connectors.rss import RSSConnector
from portolan.connectors.salesforce import SalesforceConnector
from portolan.connectors.tickets import ExportTicketConnector, ZendeskConnector
from portolan.connectors.zoom import ZoomConnector


def test_registry_maps_every_expected_name_to_its_class():
    assert CONNECTOR_REGISTRY == {
        "salesforce": SalesforceConnector,
        "hubspot": HubSpotConnector,
        "gong": GongConnector,
        "zoom": ZoomConnector,
        "clari": ClariConnector,
        "rss": RSSConnector,
        "zendesk": ZendeskConnector,
        "ticket_export": ExportTicketConnector,
    }


def test_get_connector_class_returns_the_right_class():
    assert get_connector_class("gong") is GongConnector


def test_get_connector_class_raises_a_helpful_key_error_for_unknown_names():
    with pytest.raises(KeyError, match="unknown connector 'not-a-connector'"):
        get_connector_class("not-a-connector")


def test_every_registered_class_declares_the_connector_protocol_name_attribute():
    for _name, cls in CONNECTOR_REGISTRY.items():
        assert hasattr(cls, "name"), f"{cls} has no `name` attribute"
