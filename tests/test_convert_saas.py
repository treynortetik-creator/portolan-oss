"""Tests for `scripts/data/convert_saas.py`'s pure conversion logic
(`render_call`, `render_account`, `short_id`) -- no network, fake records
only.

`scripts/` isn't a Python package (no `__init__.py`) and is excluded from
the container image, so this loads the module by file path and skips when
absent -- the same pattern `tests/test_set_owners.py` uses for
`scripts/set_owners.py`.
"""

from __future__ import annotations

import importlib.util
import json
import re
from types import ModuleType

from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "convert_saas.py")


def _load_convert_saas() -> ModuleType:
    spec = importlib.util.spec_from_file_location("convert_saas_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _conversation(*turns: tuple[str, str]) -> str:
    return json.dumps([{"speaker": speaker, "message": message} for speaker, message in turns])


def _record(**overrides: object) -> dict:
    base = {
        "conversation_id": "saas-7-conv-0",
        "company_id": "saas-7",
        "company_name": "AutoTest Pro",
        "conversation": _conversation(
            ("customer", "hi, what's the pricing?"),
            ("sales_rep", "it's $99/month"),
        ),
    }
    base.update(overrides)
    return base


_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] ([^:]+): (.*)$")


def test_render_call_produces_call_transcript_shape() -> None:
    module = _load_convert_saas()
    text = module.render_call(_record())
    lines = [line for line in text.split("\n\n") if line.strip()]
    assert len(lines) == 2

    m0 = _LINE_RE.match(lines[0])
    assert m0 is not None
    assert m0.group(1) == "00:00:00"
    assert m0.group(2) == "Customer"
    assert m0.group(3) == "hi, what's the pricing?"

    m1 = _LINE_RE.match(lines[1])
    assert m1 is not None
    assert m1.group(1) == "00:00:30"
    assert m1.group(2) == "Sales Rep"
    assert m1.group(3) == "it's $99/month"


def test_render_call_maps_system_speaker() -> None:
    module = _load_convert_saas()
    record = _record(
        conversation=_conversation(
            ("customer", "hi"),
            ("system", "--- a week later ---"),
            ("sales_rep", "following up"),
        )
    )
    text = module.render_call(record)
    assert "System: --- a week later ---" in text


def test_render_call_skips_blank_messages() -> None:
    module = _load_convert_saas()
    record = _record(
        conversation=_conversation(
            ("customer", "hi"),
            ("sales_rep", "   "),
            ("customer", "hello?"),
        )
    )
    text = module.render_call(record)
    turns = [line for line in text.split("\n\n") if line.strip()]
    assert len(turns) == 2


def test_render_call_returns_none_on_bad_json() -> None:
    module = _load_convert_saas()
    assert module.render_call(_record(conversation="not json")) is None


def test_render_call_returns_none_when_no_turns_have_text() -> None:
    module = _load_convert_saas()
    record = _record(conversation=_conversation(("customer", ""), ("sales_rep", "  ")))
    assert module.render_call(record) is None


def test_render_account_uses_company_name_when_present() -> None:
    module = _load_convert_saas()
    assert module.render_account(_record()) == "AutoTest Pro"


def test_render_account_falls_back_to_company_id() -> None:
    module = _load_convert_saas()
    record = _record(company_name="")
    assert module.render_account(record) == "synthetic-account-saas-7"


def test_render_account_falls_back_to_conversation_id_as_last_resort() -> None:
    module = _load_convert_saas()
    record = _record(company_name="", company_id="")
    assert module.render_account(record) == "synthetic-account-saas-7-conv-0"


def test_short_id_strips_redundant_saas_prefix() -> None:
    module = _load_convert_saas()
    assert module.short_id("saas-7-conv-0") == "7-conv-0"


def test_short_id_leaves_non_prefixed_ids_alone() -> None:
    module = _load_convert_saas()
    assert module.short_id("conv-42") == "conv-42"
