"""Polite HTTP for external signals: robots.txt, one request per host at a time
with a gap, a size cap, and an honest User-Agent.

Every feed, page and robots.txt request goes through `PoliteFetcher.get`, over
the same injectable `Transport` the other connectors use, so tests never touch
the network.

robots.txt (RFC 9309, https://www.rfc-editor.org/rfc/rfc9309.html) is fetched
once per host per run with our own User-Agent and read with the standard
library's parser:

- 200: its rules apply, including a `Crawl-delay`, which only ever lengthens the
  gap between requests to that host.
- 401 or 403: treated as "disallow everything" (the standard library's choice,
  stricter than RFC 9309's "unavailable").
- any other 4xx: no rules, everything allowed (RFC 9309 section 2.3.1.3).
- 5xx or no answer: disallow everything (RFC 9309 section 2.3.1.4).

The rule covers feeds too, not only pages: a feed host whose robots.txt says no
(news.google.com's does for `/rss`, checked 2026-09-23) is refused.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from portolan.connectors.base import ConnectorError, Transport, request_with_retry

__all__ = ["DEFAULT_HOST_GAP_SECONDS", "FetchResult", "PoliteFetcher", "RobotsDisallowed"]

DEFAULT_HOST_GAP_SECONDS = 5.0
# A page or feed bigger than this is cut, never parsed whole.
MAX_BODY_BYTES = 3_000_000


class RobotsDisallowed(Exception):
    """The host's robots.txt does not allow this URL for our User-Agent."""


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int
    text: str
    truncated: bool


@dataclass
class _HostRules:
    parser: RobotFileParser | None  # None: no rules (everything allowed)
    disallow_all: bool
    crawl_delay: float | None


class PoliteFetcher:
    def __init__(
        self,
        transport: Transport,
        *,
        user_agent: str,
        host_gap_seconds: float = DEFAULT_HOST_GAP_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transport = transport
        self.user_agent = user_agent
        self._host_gap = host_gap_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._rules: dict[str, _HostRules] = {}
        self._last_request: dict[str, float] = {}
        self.requests = 0

    @property
    def _agent_token(self) -> str:
        """The product token robots.txt groups match against (`PortolanSignals`)."""
        return self.user_agent.split("/", 1)[0].split(" ", 1)[0]

    def _wait_for_host(self, host: str) -> None:
        rules = self._rules.get(host)
        gap = self._host_gap
        if rules is not None and rules.crawl_delay:
            gap = max(gap, rules.crawl_delay)
        last = self._last_request.get(host)
        if last is not None:
            remaining = gap - (self._monotonic() - last)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request[host] = self._monotonic()

    def _raw_get(self, url: str) -> tuple[int, str, bool]:
        host = (urlsplit(url).hostname or "").lower()
        self._wait_for_host(host)
        self.requests += 1
        response = request_with_retry(
            self._transport,
            "GET",
            url,
            headers={"User-Agent": self.user_agent, "Accept": "*/*"},
            sleep=self._sleep,
            vendor="signals",
        )
        body = response.content or (response.text or "").encode("utf-8")
        truncated = len(body) > MAX_BODY_BYTES
        text = body[:MAX_BODY_BYTES].decode("utf-8", errors="replace")
        return response.status_code, text, truncated

    def _host_rules(self, url: str) -> _HostRules:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        cached = self._rules.get(host)
        if cached is not None:
            return cached
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        try:
            status, text, _truncated = self._raw_get(robots_url)
            rules = self._rules_from(status, text)
        except ConnectorError as exc:
            status = exc.status_code or 0
            if 400 <= status < 500 and status not in (401, 403):
                rules = _HostRules(parser=None, disallow_all=False, crawl_delay=None)
            else:
                rules = _HostRules(parser=None, disallow_all=True, crawl_delay=None)
        except Exception:
            rules = _HostRules(parser=None, disallow_all=True, crawl_delay=None)
        self._rules[host] = rules
        return rules

    def _rules_from(self, status: int, text: str) -> _HostRules:
        parser = RobotFileParser()
        parser.parse(text.splitlines())
        delay = parser.crawl_delay(self._agent_token)
        return _HostRules(
            parser=parser, disallow_all=False, crawl_delay=float(delay) if delay else None
        )

    def allowed(self, url: str) -> bool:
        rules = self._host_rules(url)
        if rules.disallow_all:
            return False
        if rules.parser is None:
            return True
        return rules.parser.can_fetch(self._agent_token, url)

    def get(self, url: str) -> FetchResult:
        """GET `url` if robots.txt allows it; raises `RobotsDisallowed` if not,
        and `ConnectorError` for a non-2xx answer."""
        if not self.allowed(url):
            raise RobotsDisallowed(f"robots.txt of {urlsplit(url).hostname} disallows {url}")
        status, text, truncated = self._raw_get(url)
        return FetchResult(url=url, status=status, text=text, truncated=truncated)
