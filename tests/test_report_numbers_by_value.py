"""The chat answer's numbers guard compares numbers by value with the one
shared normalizer, both ways: forms that restate a cited number pass, and
numbers the cited text never states still fail.

The pass cases are the shapes that made the old exact-string guard reject
correct facts in the ERB exam (01-spec/bench-erb-search-2026-09-22.md,
"Citation gate"): a number glued to its unit, a range, a number behind `~`
or `<`. The sentences here are written for this test, not taken from ERB.
"""

from __future__ import annotations

import pytest

from portolan.extract import numerals
from portolan.report.numbers import unsupported_cited_numerals
from portolan.training import overlap_gate

RESTATED = [
    pytest.param("Median latency was 250 ms.", ["p50 latency: 250ms"], id="unit-glued-in-source"),
    pytest.param("Median latency was 250ms.", ["p50 latency is 250 ms"], id="unit-glued-in-claim"),
    pytest.param("Batches take 20 to 40 minutes.", ["batch time 20–40 min"], id="range-en-dash"),
    pytest.param("Batches take 20-40 minutes.", ["batch time 20–40 min"], id="range-both-sides"),
    pytest.param("Retries peak at about 200 per hour.", ["retries ~200/hr"], id="tilde"),
    pytest.param("Error rate stays under 5%.", ["error rate <5%"], id="less-than"),
    pytest.param("The plan costs $5,000 a month.", ["It's $5k/month."], id="k-suffix"),
    pytest.param("The fund needs $30 million.", ["needs approximately $30M"], id="m-suffix"),
    pytest.param("They signed a 5-year deal.", ["a 5 year agreement"], id="hyphen-unit"),
    pytest.param("They raised $3M.", ["raised $3,000,000"], id="currency-scale"),
    pytest.param("They raised $3 million.", ["raised $3M"], id="scale-word-from-suffix"),
    pytest.param("Usage rose 37-percent.", ["usage up 37%"], id="hyphen-percent"),
    pytest.param("Usage rose 37%.", ["usage up 37 percent"], id="percent-word"),
    pytest.param("Usage rose 37 per cent.", ["37-percent growth"], id="per-cent"),
    pytest.param("Usage rose 5%.", ["usage rose five percent"], id="spelled-percent"),
    pytest.param("CPU sat at 25-30%.", ["cpu 25 -30%"], id="percent-range"),
]

INVENTED = [
    pytest.param("Median latency was 300 ms.", ["p50 latency: 250ms"], ["300"], id="other-value"),
    pytest.param("Median latency was 900ms.", ["p50 latency: 250ms"], ["900"], id="glued-invented"),
    pytest.param(
        "Batches take 20 to 50 minutes.", ["batch time 20–40 min"], ["50"], id="range-end-changed"
    ),
    pytest.param("The plan costs $5,500 a month.", ["It's $5k/month."], ["5,500"], id="near-miss"),
    pytest.param("The fund needs $300 million.", ["needs $30M"], ["300 million"], id="scale-x10"),
    pytest.param("They signed a 6-year deal.", ["a 5 year agreement"], ["6-year"], id="unit-value"),
    pytest.param(
        "They signed a 5-month deal.", ["a 5 year agreement"], ["5-month"], id="unit-changed"
    ),
    pytest.param("About 5 million users.", ["About 5 users"], ["5 million"], id="scale-word"),
    pytest.param("Retries peak at 41.", ["no figures here"], ["41"], id="nothing-cited"),
    # 3 million is not 3; a percent is not a count
    pytest.param("They raised $3M.", ["They bought 3 pumps."], ["3M"], id="currency-m-vs-count"),
    pytest.param("They raised 3M.", ["They bought 3 pumps."], ["3M"], id="capital-m-vs-count"),
    pytest.param(
        "They raised 3 million.", ["They bought 3 pumps."], ["3 million"], id="million-vs-count"
    ),
    pytest.param(
        "Usage rose 37-percent.", ["37 pumps were installed"], ["37-percent"], id="hyphen-percent"
    ),
    pytest.param("Usage rose 37-percent.", ["no figures"], ["37-percent"], id="percent-unstated"),
    pytest.param("Usage rose 37%.", ["37 pumps were installed"], ["37%"], id="percent-vs-count"),
    pytest.param("37 pumps were installed.", ["usage rose 37%"], ["37"], id="count-vs-percent"),
]


@pytest.mark.parametrize(("sentence", "cited"), RESTATED)
def test_a_restated_number_passes_by_value(sentence: str, cited: list[str]) -> None:
    assert unsupported_cited_numerals(sentence, cited) == []


@pytest.mark.parametrize(("sentence", "cited", "missing"), INVENTED)
def test_an_invented_number_still_fails(
    sentence: str, cited: list[str], missing: list[str]
) -> None:
    assert unsupported_cited_numerals(sentence, cited) == missing


@pytest.mark.parametrize(
    "sentence",
    [
        "They complained on 2026-08-04.",
        "They complained on August 4, 2026.",
        "They complained in 2025.",
        "The 3rd renewal is ticket SUP-100 on account A-1010.",
        "The call started at 14:30.",
        "Filed 09/30/2026 ^[S-call-007 @ 00:14:32] per [[ashgrove-q3-2026]].",
    ],
)
def test_dates_years_ids_ordinals_and_markers_are_never_required(sentence: str) -> None:
    assert unsupported_cited_numerals(sentence, ["no figures here"]) == []


def test_a_number_may_come_from_any_cited_text() -> None:
    assert unsupported_cited_numerals("47 pumps and 3 sites.", ["47 pumps", "at 3 sites"]) == []
    assert unsupported_cited_numerals("47 pumps and 4 sites.", ["47 pumps", "at 3 sites"]) == ["4"]


def test_the_chat_guard_and_the_silver_gate_share_one_normalizer() -> None:
    assert overlap_gate.unsupported_numerals is numerals.unsupported_numerals
