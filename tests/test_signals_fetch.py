"""`portolan.signals.fetch.PoliteFetcher`: robots.txt, per-host gaps, the User-Agent.
Every response is served by `FakeTransport`; nothing touches the network."""

from __future__ import annotations

import pytest
from connector_helpers import FakeTransport, text_response

from portolan.signals.fetch import PoliteFetcher, RobotsDisallowed

UA = "PortolanSignals/0.1 (test)"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _fetcher(transport: FakeTransport, clock: _Clock | None = None, gap: float = 5.0):
    clock = clock or _Clock()
    return (
        PoliteFetcher(
            transport,
            user_agent=UA,
            host_gap_seconds=gap,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        ),
        clock,
    )


def test_disallowed_path_is_refused_without_fetching_it() -> None:
    transport = FakeTransport()
    transport.queue(
        "https://ex.org/robots.txt", text_response("User-agent: *\nDisallow: /private/\n")
    )
    fetcher, _ = _fetcher(transport)

    with pytest.raises(RobotsDisallowed):
        fetcher.get("https://ex.org/private/page")
    assert [c["url"] for c in transport.calls] == ["https://ex.org/robots.txt"]


def test_group_for_our_agent_wins_over_the_star_group() -> None:
    transport = FakeTransport()
    transport.queue(
        "https://ex.org/robots.txt",
        text_response("User-agent: PortolanSignals\nDisallow: /\n\nUser-agent: *\nAllow: /\n"),
    )
    fetcher, _ = _fetcher(transport)

    assert fetcher.allowed("https://ex.org/news") is False


@pytest.mark.parametrize(("status", "allowed"), [(404, True), (403, False), (500, False)])
def test_robots_status_codes(status: int, allowed: bool) -> None:
    transport = FakeTransport()
    transport.queue("https://ex.org/robots.txt", text_response("", status_code=status))
    fetcher, _ = _fetcher(transport)

    assert fetcher.allowed("https://ex.org/news") is allowed


def test_robots_is_read_once_per_host_and_the_user_agent_is_sent() -> None:
    transport = FakeTransport()
    transport.queue("https://ex.org/robots.txt", text_response("User-agent: *\nAllow: /\n"))
    transport.queue("https://ex.org/a", text_response("<p>a</p>"))
    transport.queue("https://ex.org/b", text_response("<p>b</p>"))
    fetcher, _ = _fetcher(transport)

    assert fetcher.get("https://ex.org/a").text == "<p>a</p>"
    fetcher.get("https://ex.org/b")

    urls = [c["url"] for c in transport.calls]
    assert urls.count("https://ex.org/robots.txt") == 1
    assert all(c["headers"]["User-Agent"] == UA for c in transport.calls)


def test_requests_to_one_host_wait_for_the_gap_and_crawl_delay_lengthens_it() -> None:
    transport = FakeTransport()
    transport.queue(
        "https://slow.org/robots.txt", text_response("User-agent: *\nCrawl-delay: 10\n")
    )
    transport.queue("https://slow.org/a", text_response("a"))
    transport.queue("https://slow.org/b", text_response("b"))
    fetcher, clock = _fetcher(transport, gap=5.0)

    fetcher.get("https://slow.org/a")
    fetcher.get("https://slow.org/b")

    # robots.txt is read first; every later request to the host, /a included,
    # waits the 10 s crawl delay rather than the 5 s default.
    assert clock.slept == [10.0, 10.0]


def test_different_hosts_do_not_wait_for_each_other() -> None:
    transport = FakeTransport()
    for host in ("one.org", "two.org"):
        transport.queue(f"https://{host}/robots.txt", text_response("", status_code=404))
        transport.queue(f"https://{host}/a", text_response("a"))
    fetcher, clock = _fetcher(transport, gap=5.0)

    fetcher.get("https://one.org/a")
    fetcher.get("https://two.org/a")

    # One gap per host (robots.txt -> /a), none between the hosts.
    assert clock.slept == [5.0, 5.0]
