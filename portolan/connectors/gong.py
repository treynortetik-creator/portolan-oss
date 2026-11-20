"""Gong connector (build-contracts.md "Connectors": Gong).

Docs relied on (researched 2026-09-12; Gong's own API documentation lives
behind an authenticated workspace login at
`https://<workspace>.app.gong.io/settings/api/documentation` and returned
no fetchable content, so the shapes below come from Gong's public
developer write-ups and third-party clients that mirror the documented
v2 API):
- Basic auth with an access key and secret:
  https://www.claap.io/blog/gong-api
- `GET /v2/calls` with `fromDateTime` / `toDateTime` (ISO-8601, capped at
  a 90-day window per request) and cursor pagination via
  `records.cursor` in the response, passed back as the `cursor` query
  param on the next request:
  https://www.claap.io/blog/gong-api ;
  https://github.com/ksindi/gong-client/blob/main/docs/CallsApi.md
- `POST /v2/calls/transcript` with a `filter.callIds` body, returning
  `callTranscripts: [{callId, transcript: [{speakerId, topic, sentences:
  [{start, end, text}]}]}]`, `start`/`end` in milliseconds from the start
  of the call: same sources, plus the Gong community thread "Retrieve
  call transcript through API using python".
- Call parties (participant name/email per speaker) come from
  `POST /v2/calls/extensive` with `contentSelector.exposedFields.parties:
  true`, returning `parties: [{id, name, emailAddress, speakerId}]` per
  call: same third-party write-ups.

Not independently verified against a live Gong workspace: the exact
top-level key name of the calls list response (`calls` is used here,
consistent across every secondary source found) and whether a transcript
sentence's `speakerId` always has a matching `parties[].speakerId`
(Gong's own docs note it can be absent for cross-talk or an
unidentified speaker; handled here by falling back to "Unknown
Speaker").
"""

from __future__ import annotations

import base64
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
    iter_date_windows,
    parse_datetime,
    request_with_retry,
    source_day,
)
from portolan.extract.schema import SourceType
from portolan.ingest.models import Participant, RawSource
from portolan.util import clock

__all__ = ["GongConfig", "GongConnector"]

# `/v2/calls` caps `fromDateTime`..`toDateTime` at 90 days (module docstring).
_GONG_MAX_WINDOW_DAYS = 90


@dataclass(frozen=True)
class GongConfig:
    """A Gong API access key/secret pair, never from code."""

    access_key: str
    access_key_secret: str
    base_url: str = "https://api.gong.io"


def _find_call_transcript(transcript_payload: dict, call_id: str) -> list[dict]:
    for entry in transcript_payload.get("callTranscripts", []):
        if str(entry.get("callId")) == str(call_id):
            return entry.get("transcript", [])
    return []


def _extract_parties(extensive_payload: dict, call_id: str) -> list[dict]:
    for call in extensive_payload.get("calls", []):
        if str(call.get("metaData", {}).get("id") or call.get("id")) == str(call_id):
            return call.get("parties", [])
    # Some accounts return parties flattened at the top level for a
    # single-call request rather than nested under `calls`.
    return extensive_payload.get("parties", [])


def _render_transcript(transcript: list[dict], speaker_names: dict[str, str]) -> str:
    lines: list[str] = []
    for utterance in transcript:
        sentences = utterance.get("sentences") or []
        if not sentences:
            continue
        speaker_name = speaker_names.get(utterance.get("speakerId"), "Unknown Speaker")
        timestamp = format_ms_offset(sentences[0].get("start", 0))
        text = " ".join(s.get("text", "").strip() for s in sentences if s.get("text"))
        lines.append(f"[{timestamp}] {speaker_name}: {text}")
    return "\n".join(lines)


class GongConnector:
    """`build-contracts.md`'s `Connector` protocol over Gong's v2 API.
    `list_sources` hits `/v2/calls` only; `fetch` makes the two
    follow-up calls a call's full text needs (transcript, parties)."""

    name = "gong"

    def __init__(
        self,
        config: GongConfig,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or httpx_transport(httpx.Client(timeout=30.0))
        self._sleep = sleep

    def _headers(self) -> dict[str, str]:
        token = base64.b64encode(
            f"{self._config.access_key}:{self._config.access_key_secret}".encode()
        ).decode()
        return {"Authorization": f"Basic {token}"}

    def list_sources(self, since: date | None) -> Iterable[SourceRef]:
        """`since=None` lets Gong's own default answer; a `since` is split
        into windows of at most `_GONG_MAX_WINDOW_DAYS`, the last one open
        to now. A call listed by two windows is yielded once."""
        if since is None:
            yield from self._list_calls_window({})
            return
        seen: set[str] = set()
        windows = list(iter_date_windows(since, clock.today(), _GONG_MAX_WINDOW_DAYS))
        for index, (window_from, window_to) in enumerate(windows):
            params: dict[str, object] = {"fromDateTime": f"{window_from.isoformat()}T00:00:00Z"}
            if index < len(windows) - 1:
                params["toDateTime"] = f"{window_to.isoformat()}T00:00:00Z"
            for ref in self._list_calls_window(params):
                if ref.source_id not in seen:
                    seen.add(ref.source_id)
                    yield ref

    def _list_calls_window(self, params: dict[str, object]) -> Iterable[SourceRef]:
        cursor: str | None = None
        while True:
            request_params = dict(params, cursor=cursor) if cursor else dict(params)
            response = request_with_retry(
                self._transport,
                "GET",
                f"{self._config.base_url}/v2/calls",
                params=request_params,
                headers=self._headers(),
                sleep=self._sleep,
                vendor="gong",
            )
            payload = response.json()
            for call in payload.get("calls", []):
                started = call.get("started") or call.get("scheduled")
                yield SourceRef(
                    source_id=f"gong-{call['id']}",
                    source_type=SourceType.CALL,
                    updated_at=parse_datetime(started),
                    ref={"call": call},
                )
            cursor = (payload.get("records") or {}).get("cursor")
            if not cursor:
                break

    def fetch(self, ref: SourceRef) -> RawSource:
        call = ref.ref["call"]
        call_id = call["id"]

        transcript_payload = request_with_retry(
            self._transport,
            "POST",
            f"{self._config.base_url}/v2/calls/transcript",
            json={"filter": {"callIds": [call_id]}},
            headers=self._headers(),
            sleep=self._sleep,
            vendor="gong",
        ).json()

        extensive_payload = request_with_retry(
            self._transport,
            "POST",
            f"{self._config.base_url}/v2/calls/extensive",
            json={
                "filter": {"callIds": [call_id]},
                "contentSelector": {"exposedFields": {"parties": True}},
            },
            headers=self._headers(),
            sleep=self._sleep,
            vendor="gong",
        ).json()

        parties = _extract_parties(extensive_payload, call_id)
        speaker_names = {
            p["speakerId"]: p.get("name", "Unknown Speaker") for p in parties if p.get("speakerId")
        }
        transcript = _find_call_transcript(transcript_payload, call_id)
        text = _render_transcript(transcript, speaker_names)

        participants = [
            Participant(name=p.get("name") or "Unknown", email=p.get("emailAddress"))
            for p in parties
        ]

        started = call.get("started") or call.get("scheduled")
        source_date = source_day(parse_datetime(started)) if started else clock.today()

        return RawSource(
            source_id=f"gong-{call_id}",
            source_type=SourceType.CALL,
            source_date=source_date,
            participants=participants,
            account_hints=[],
            text=text,
            original_bytes=json.dumps(
                {"call": call, "transcript": transcript_payload, "extensive": extensive_payload},
                sort_keys=True,
            ).encode("utf-8"),
            original_content_type="application/json",
            metadata={"call_id": call_id, "title": call.get("title")},
        )
