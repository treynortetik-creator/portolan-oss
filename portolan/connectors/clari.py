"""Clari Copilot (formerly Wingman) connector.

**EXPERIMENTAL.** Clari Copilot's public API reference at
`https://api-doc.copilot.clari.com/` returned only a page title with no
endpoint content when fetched on 2026-09-12 (it appears to be gated
behind a login or rendered client-side). The shapes below are read from
Clari's community forum instead (`community.clari.com`, searched
2026-09-12: "How To Use Copilot APIs", "Clari API — All You Need To
Know") plus its own note that "Authentication uses X-Api-Key + Password,
with a 10 req/sec rate limit" and that "the /call-details endpoint
provides transcripts with speaker turns", keyed by a call id.

**Needs verification against a real account before go-live:**
- The list-conversations endpoint path and its pagination shape. No
  public source names one; this connector assumes `GET
  /api/v1/conversations` with `startDate` and a `nextPageToken` cursor,
  mirroring Gong's and Zoom's conventions (the same category of vendor).
- The exact field names on each `/call-details` speaker turn. Community
  sources confirm the endpoint returns "transcripts with speaker turns"
  but not the field names; this connector assumes `speakerName`,
  `speakerEmail`, `startTime` (milliseconds from call start, Gong's
  convention), and `text`, under `turns` or `transcript`.
- The header names for the two-part API key/password auth. Assumed
  `X-Api-Key` and `X-Api-Password` here, following the community note's
  wording; a real account's response to a bad-credentials request would
  confirm this immediately.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date

import httpx

from portolan.connectors.base import (
    SourceRef,
    Transport,
    format_ms_offset,
    httpx_transport,
    parse_datetime,
    request_with_retry,
    source_day,
)
from portolan.extract.schema import SourceType
from portolan.ingest.models import Participant, RawSource
from portolan.util import clock

__all__ = ["ClariConfig", "ClariConnector"]


@dataclass(frozen=True)
class ClariConfig:
    """An API key/password pair, never from code."""

    api_key: str
    api_password: str
    base_url: str = "https://api.clari.com"


class ClariConnector:
    """`build-contracts.md`'s `Connector` protocol over Clari Copilot's
    API, EXPERIMENTAL (module docstring)."""

    name = "clari"

    def __init__(
        self,
        config: ClariConfig,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or httpx_transport(httpx.Client(timeout=30.0))
        self._sleep = sleep

    def _headers(self) -> dict[str, str]:
        return {
            "X-Api-Key": self._config.api_key,
            "X-Api-Password": self._config.api_password,
        }

    def list_sources(self, since: date | None) -> Iterable[SourceRef]:
        params: dict[str, object] = {}
        if since is not None:
            params["startDate"] = since.isoformat()
        token: str | None = None
        while True:
            request_params = dict(params, nextPageToken=token) if token else dict(params)
            response = request_with_retry(
                self._transport,
                "GET",
                f"{self._config.base_url}/api/v1/conversations",
                params=request_params,
                headers=self._headers(),
                sleep=self._sleep,
                vendor="clari",
            )
            payload = response.json()
            for conversation in payload.get("conversations", []):
                started = conversation.get("startTime") or conversation.get("date")
                yield SourceRef(
                    source_id=f"clari-{conversation['id']}",
                    source_type=SourceType.CALL,
                    updated_at=parse_datetime(started),
                    ref={"conversation": conversation},
                )
            token = payload.get("nextPageToken")
            if not token:
                break

    def fetch(self, ref: SourceRef) -> RawSource:
        conversation = ref.ref["conversation"]
        call_id = conversation["id"]

        payload = request_with_retry(
            self._transport,
            "GET",
            f"{self._config.base_url}/call-details",
            params={"callId": call_id},
            headers=self._headers(),
            sleep=self._sleep,
            vendor="clari",
        ).json()

        turns = payload.get("turns") or payload.get("transcript") or []
        lines: list[str] = []
        participants: list[Participant] = []
        seen_speakers: set[str] = set()
        for turn in turns:
            speaker = turn.get("speakerName") or "Unknown Speaker"
            if speaker not in seen_speakers:
                seen_speakers.add(speaker)
                email = turn.get("speakerEmail")
                participants.append(Participant(name=speaker, email=email))
            timestamp = format_ms_offset(turn.get("startTime", 0))
            lines.append(f"[{timestamp}] {speaker}: {(turn.get('text') or '').strip()}")

        started = conversation.get("startTime") or conversation.get("date")
        source_date = source_day(parse_datetime(started)) if started else clock.today()

        return RawSource(
            source_id=f"clari-{call_id}",
            source_type=SourceType.CALL,
            source_date=source_date,
            participants=participants,
            account_hints=[],
            text="\n".join(lines),
            original_bytes=json.dumps(
                {"conversation": conversation, "details": payload}, sort_keys=True
            ).encode("utf-8"),
            original_content_type="application/json",
            metadata={"call_id": call_id, "title": conversation.get("title")},
        )
