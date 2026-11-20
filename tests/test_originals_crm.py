"""A claim quoted from a CRM row must be on that row's original page, at the
cited locator, and `scripts/verify.py chat` must find it there.

The file connector hands the extractor a row as `key=value, ...` in the
export's column order, but stores the row as JSON with sorted keys, so the
viewer cannot show the extractor's string: it shows the row as one block at
its locator, one `key=value` line per field. The quotes below are the live
graph's (claims 526, 529 and 530 on 2026-09-23), which `verify.py chat`
could not find on the pretty-printed JSON the viewer used to show.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from urllib.parse import quote as urlquote

import pytest
from fastapi.testclient import TestClient

from portolan.config import Settings
from portolan.connectors.file_connector import FileConnector
from portolan.ingest.intake import intake
from portolan.store import LocalStore
from portolan.web.app import create_app
from portolan.web.deps import get_conn, get_store_dep
from tests._paths import require_repo_path

CORPUS_OUT = Path(__file__).resolve().parent.parent / "corpus" / "out"
ORG_KEY = "test-org-key-for-crm-originals"

ASHGROVE_V2 = "S-crm-opp-A1010-v2"
ASHGROVE_V2_AT = "crm/opportunities-2026-08-21.csv#O-3001"
ASHGROVE_V2_QUOTE = (
    "opportunity_id=O-3001, account_crm_id=A-1010, name=Ashgrove Health — HalvardCare "
    "renewal, stage=negotiation, close_date=2027-06-30, amount=2350000, export_date=2026-08-21"
)
NORTHLAKE = "S-crm-opp-A1030"
NORTHLAKE_AT = "crm/opportunities.csv#O-3003"
NORTHLAKE_QUOTE = (
    "opportunity_id=O-3003, account_crm_id=A-1030, name=Northlake Health System — Tidewater "
    "T3 tower expansion, stage=qualification, close_date=2027-01-31, amount=300000, "
    "export_date=2026-04-15"
)


def _load_verify():
    path = require_repo_path("scripts", "verify.py")
    spec = importlib.util.spec_from_file_location("verify_crm_originals", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_crm_originals"] = module
    spec.loader.exec_module(module)
    return module


verify = _load_verify()


@pytest.fixture()
def crm_client(db, db_url: str, tmp_path: Path):
    """The two live CRM sources, taken in through the real connector and
    `intake` (stored row bytes and connector metadata), behind the app."""
    store = LocalStore(tmp_path / "store")
    connector = FileConnector(CORPUS_OUT)
    refs = {r.source_id: r for r in connector.list_sources() if r.source_id.startswith("S-crm")}
    for source_id in (ASHGROVE_V2, NORTHLAKE):
        intake(connector.fetch(refs[source_id]), conn=db, store=store)

    settings = Settings(
        database_url=db_url,
        tenant="test",
        store_root=tmp_path / "store",
        llm_provider="fake",
        embed_provider="fake",
        org_key=ORG_KEY,
        session_secret="test-session-secret-for-crm-originals-only",
        cookie_secure=False,
    )
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_store_dep] = lambda: store
    with TestClient(app) as client:
        login = client.post(
            "/login", data={"org_key": ORG_KEY, "editor": "Test Editor"}, follow_redirects=False
        )
        assert login.status_code == 303
        yield client


def _check(client: TestClient, slug: str, at: str, claim_quote: str) -> dict:
    """`verify.py chat`'s check of one citation's original, as it runs live."""
    href = f"/originals/{slug}?at={urlquote(at, safe='/')}"
    response = client.get(href)
    assert response.status_code == 200
    citation = {"original_href": href, "quote": claim_quote}
    problems: list[str] = []
    verify._check_original(citation, verify.Page(response), problems)
    return {**citation, "problems": problems, "html": response.text}


@pytest.mark.parametrize(
    ("slug", "at", "claim_quote"),
    [(ASHGROVE_V2, ASHGROVE_V2_AT, ASHGROVE_V2_QUOTE), (NORTHLAKE, NORTHLAKE_AT, NORTHLAKE_QUOTE)],
)
def test_a_crm_rows_quote_is_on_its_original_at_the_cited_row(crm_client, slug, at, claim_quote):
    result = _check(crm_client, slug, at, claim_quote)

    assert result["problems"] == []
    assert result["cited_block_found"] is True
    assert result["quote_in_cited_block"] is True
    assert result["quote_match"] == "fields"
    # Every field the quote states is on the page in the quote's own form.
    pairs = re.split(r", (?=[a-z_]+=)", claim_quote)
    assert len(pairs) == 7
    for pair in pairs:
        key, value = pair.split("=", 1)
        field = f'<span class="crm-key">{key}</span>=<span class="crm-value">{value}</span>'
        assert field in result["html"]


def test_a_quote_from_another_snapshot_of_the_row_is_not_found(crm_client):
    """The April export's close date is not on the August snapshot's page."""
    april = ASHGROVE_V2_QUOTE.replace("2027-06-30", "2027-03-31").replace(
        "export_date=2026-08-21", "export_date=2026-04-15"
    )
    result = _check(crm_client, ASHGROVE_V2, ASHGROVE_V2_AT, april)

    assert result["quote_match"] == "none"
    assert result["problems"] == [f"the claim's quote is not on {result['original_href']}"]


def test_row_quote_match_reads_each_connector_form_and_nothing_else():
    fields = {"Name": "Big Deal, Inc.", "StageName": "Closed Won", "Amount": "50000"}
    match = verify.row_quote_match

    assert match("Opportunity: Big Deal, Inc.\nStageName: Closed Won", fields, "opportunity") == (
        "fields"
    )
    assert match("amount=50000, name=Big Deal, Inc.", fields) == "fields"
    assert match("Closed Won", fields) == "fields"
    assert match("StageName: Closed Lost", fields) == "none"
    assert match("amount=500001", fields) == "none"
    assert match("StageName: Closed Won, and the buyer is unhappy", fields) == "none"
    assert match("", fields) == "none"
