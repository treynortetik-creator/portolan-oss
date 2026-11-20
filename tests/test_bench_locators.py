"""Tests for `portolan.bench.locators`, focused on the diagnosis-driven
fix: a real model's locator often carries the source text's own enclosing
brackets, and sometimes a range rather than a single point in time."""

from __future__ import annotations

from portolan.bench.locators import (  # noqa: PLC2701 - _rss_url unit-tested directly
    _rss_url,
    locator_matches,
    parse_hhmmss,
    trailing_int,
)


def test_parse_hhmmss_plain() -> None:
    assert parse_hhmmss("00:14:32") == 14 * 60 + 32


def test_parse_hhmmss_strips_enclosing_brackets() -> None:
    """Diagnosis: a real produced fact's locator was `[00:05:26]`, brackets
    and all -- copied straight from the source text's own `[HH:MM:SS]`
    prefix rather than just the digits inside it."""
    assert parse_hhmmss("[00:14:32]") == 14 * 60 + 32


def test_parse_hhmmss_takes_the_first_timestamp_in_a_range() -> None:
    """Diagnosis: a real produced fact's locator was `[00:04:30-00:04:43]`
    -- a range over the lines the fact was synthesized from. The start of
    the range is the comparable point in time."""
    assert parse_hhmmss("[00:04:30-00:04:43]") == 4 * 60 + 30
    assert parse_hhmmss("00:04:30-00:04:43") == 4 * 60 + 30


def test_parse_hhmmss_none_for_non_timestamp() -> None:
    assert parse_hhmmss("S-ticket-031#1") is None
    assert parse_hhmmss("¶2") is None
    assert parse_hhmmss("") is None


def test_trailing_int_strips_enclosing_brackets() -> None:
    assert trailing_int("[S-ticket-031#1]") == 1
    assert trailing_int("[¶2]") == 2


def test_locator_matches_call_within_window_despite_brackets_and_range() -> None:
    assert locator_matches("call", "00:05:20", "[00:05:20-00:05:34]")
    assert locator_matches("call", "00:13:45", "[00:13:45-00:14:11]")
    # Still correctly rejects a timestamp outside the 90-second window.
    assert not locator_matches("call", "00:05:20", "[00:20:00-00:20:15]")


def test_locator_matches_ticket_and_email_despite_brackets() -> None:
    assert locator_matches("ticket", "S-ticket-031#1", "[S-ticket-031#1]")
    assert locator_matches("ticket", "S-ticket-031#1", "[1]")
    assert locator_matches("email", "¶2", "[¶2]")
    assert locator_matches("email", "¶2", "[¶3]")  # within +/- 1
    assert not locator_matches("email", "¶2", "[¶5]")


# --------------------------------------------------------------------------
# RSS locator fix (2026-09-18 harness review, 01-spec/harness-review-
# 2026-09-18.md): the truth table's planted locator is `url:<link> <date>`
# (corpus_loader.py's own `[url:<link> <date>]` line), but the extraction
# prompt tells the model the locator is "the line's own url" -- url only.
# The old rule (`planted.lower() == produced.lower()`) failed every
# url-only produced locator against the date-bearing planted one, even
# though the model did exactly what the prompt asked.
_PLANTED_RSS_LOCATOR = (
    "url:https://feeds.example-newswire.test/portolan-corpus/s-rss-006 2026-09-02"
)


def test_rss_url_strips_the_date_and_url_prefix() -> None:
    assert (
        _rss_url(_PLANTED_RSS_LOCATOR)
        == "https://feeds.example-newswire.test/portolan-corpus/s-rss-006"
    )
    assert _rss_url("https://feeds.example-newswire.test/portolan-corpus/s-rss-006") == (
        "https://feeds.example-newswire.test/portolan-corpus/s-rss-006"
    )


def test_locator_matches_rss_url_only_produced_locator() -> None:
    """A model following the prompt's "the line's own url" instruction to
    the letter -- no date -- now matches the date-bearing planted locator."""
    assert locator_matches(
        "rss",
        _PLANTED_RSS_LOCATOR,
        "https://feeds.example-newswire.test/portolan-corpus/s-rss-006",
    )


def test_locator_matches_rss_tolerates_url_prefix_brackets_and_case() -> None:
    assert locator_matches(
        "rss",
        _PLANTED_RSS_LOCATOR,
        "URL:https://feeds.example-newswire.test/portolan-corpus/S-RSS-006",
    )
    assert locator_matches(
        "rss",
        _PLANTED_RSS_LOCATOR,
        "[url:https://feeds.example-newswire.test/portolan-corpus/s-rss-006]",
    )
    # Still an exact match on the date, when the model does include one.
    assert locator_matches(
        "rss",
        _PLANTED_RSS_LOCATOR,
        "url:https://feeds.example-newswire.test/portolan-corpus/s-rss-006 2026-09-02",
    )


def test_locator_matches_rss_rejects_a_different_item() -> None:
    assert not locator_matches(
        "rss",
        _PLANTED_RSS_LOCATOR,
        "https://feeds.example-newswire.test/portolan-corpus/s-rss-004",
    )
