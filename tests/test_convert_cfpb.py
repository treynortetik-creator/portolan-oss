"""Tests for `scripts/data/convert_cfpb.py`'s pure conversion logic
(`render_ticket`) -- no network, fake records only.

`scripts/` isn't a Python package (no `__init__.py`) and is excluded from
the container image, so this loads the module by file path and skips when
absent -- the same pattern `tests/test_set_owners.py` uses for
`scripts/set_owners.py`.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "convert_cfpb.py")


def _load_convert_cfpb() -> ModuleType:
    spec = importlib.util.spec_from_file_location("convert_cfpb_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _record(**overrides: object) -> dict:
    base = {
        "complaint_id": "12345",
        "product": "Debt collection",
        "company": "Acme Collections LLC",
        "date_received": "2022-03-14",
        "complaint_what_happened": "They keep calling me about a debt that isn't mine.",
        "company_public_response": "",
    }
    base.update(overrides)
    return base


def test_render_ticket_maps_fields() -> None:
    module = _load_convert_cfpb()
    ticket = module.render_ticket(_record())
    assert ticket is not None
    assert ticket["id"] == "S-ticket-cfpb-12345"
    assert ticket["requester"] == "consumer"
    assert ticket["account"] == "Acme Collections LLC"
    assert ticket["created"] == "2022-03-14"
    assert ticket["subject"] == "Debt collection"


def test_render_ticket_narrative_is_first_comment() -> None:
    module = _load_convert_cfpb()
    ticket = module.render_ticket(_record())
    assert len(ticket["comments"]) == 1
    assert ticket["comments"][0]["number"] == 1
    assert ticket["comments"][0]["author"] == "Consumer"
    assert "keep calling me" in ticket["comments"][0]["body"]


def test_render_ticket_adds_company_response_as_second_comment_when_present() -> None:
    module = _load_convert_cfpb()
    ticket = module.render_ticket(
        _record(company_public_response="Company believes it acted appropriately")
    )
    assert len(ticket["comments"]) == 2
    assert ticket["comments"][1]["number"] == 2
    assert ticket["comments"][1]["author"] == "Acme Collections LLC"
    assert ticket["comments"][1]["body"] == "Company believes it acted appropriately"


def test_render_ticket_omits_second_comment_when_response_blank() -> None:
    module = _load_convert_cfpb()
    ticket = module.render_ticket(_record(company_public_response="   "))
    assert len(ticket["comments"]) == 1


def test_render_ticket_returns_none_without_narrative() -> None:
    module = _load_convert_cfpb()
    assert module.render_ticket(_record(complaint_what_happened="")) is None
    assert module.render_ticket(_record(complaint_what_happened="   ")) is None
    assert module.render_ticket(_record(complaint_what_happened=None)) is None


def test_render_ticket_returns_none_without_complaint_id() -> None:
    module = _load_convert_cfpb()
    assert module.render_ticket(_record(complaint_id="")) is None
    assert module.render_ticket(_record(complaint_id=None)) is None


def test_render_ticket_returns_none_without_date() -> None:
    module = _load_convert_cfpb()
    assert module.render_ticket(_record(date_received="")) is None
    assert module.render_ticket(_record(date_received=None)) is None


def test_render_ticket_falls_back_to_unknown_company() -> None:
    module = _load_convert_cfpb()
    ticket = module.render_ticket(_record(company=""))
    assert ticket is not None
    assert ticket["account"] == "Unknown Company"


def test_render_ticket_falls_back_subject_to_complaint() -> None:
    module = _load_convert_cfpb()
    ticket = module.render_ticket(_record(product=""))
    assert ticket is not None
    assert ticket["subject"] == "Complaint"
