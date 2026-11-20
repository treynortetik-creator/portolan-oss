"""`portolan.report.outliers` tests (wave C1)."""

# ruff: noqa: F811  (pytest fixtures imported by name; see test_report_helpers.py)

from __future__ import annotations

from datetime import date

import pytest
from test_report_helpers import FullGraph, build_report_llm, full_graph  # noqa: F401

from portolan.report import outliers, themes

WINDOW_START = date(2026, 3, 2)
AS_OF = date(2026, 9, 4)


@pytest.fixture(scope="module", autouse=True)
def _themes_refreshed(full_graph: FullGraph):
    """`refresh_themes` is idempotent (it only ever considers claims not
    already in `theme_members`), so running it here -- before this
    module's tests check what's *left* unattached -- is safe regardless
    of whether `tests/test_report_themes.py` has already run it in this
    same session."""
    themes.refresh_themes(
        full_graph.conn,
        build_report_llm(),
        full_graph.embedder,
        since=WINDOW_START,
        as_of=AS_OF,
    )


def test_f06_the_cfo_renewal_risk_is_an_outlier(full_graph: FullGraph) -> None:
    """F06 (`corpus/truth-table.yaml`) is a `risk`-type claim on
    `ashgrove-health` with no product entity -- it shares neither fact
    type nor product with the late-deliveries theme's founding
    `complaint`/`tidewater-t2` cluster, so it should never attach
    (`portolan.report.themes._qualifies`) and should surface here."""
    result = outliers.list_outliers(full_graph.conn, since=WINDOW_START, as_of=AS_OF)
    matches = [o for o in result if "raised at the 2027 renewal" in o.claim_text]
    assert matches, [o.claim_text for o in result]
    assert matches[0].account_slug == "ashgrove-health"
    assert matches[0].fact_type == "risk"


def test_outliers_are_all_unattached_signal_claims(full_graph: FullGraph) -> None:
    result = outliers.list_outliers(full_graph.conn, since=WINDOW_START, as_of=AS_OF)
    assert result  # the corpus plants far more signals than any one theme covers
    claim_ids = [o.claim_id for o in result]
    with full_graph.conn.cursor() as cur:
        cur.execute("SELECT claim_id FROM theme_members WHERE claim_id = ANY(%s)", (claim_ids,))
        overlap = cur.fetchall()
    assert overlap == []


def test_outliers_carry_source_and_locator_for_plain_display(full_graph: FullGraph) -> None:
    result = outliers.list_outliers(full_graph.conn, since=WINDOW_START, as_of=AS_OF)
    assert all(o.source_slug for o in result)
    assert all(o.locator for o in result)
