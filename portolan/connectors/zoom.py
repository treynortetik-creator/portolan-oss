"""Zoom connector (build-contracts.md "Connectors": Zoom).

Docs relied on (researched 2026-09-12):
- Server-to-Server OAuth: token endpoint `https://zoom.us/oauth/token`,
  `grant_type=account_credentials` + `account_id` as query/body params,
  HTTP Basic auth with the client id/secret, JSON response carrying
  `access_token` / `token_type` / `expires_in`:
  https://developers.zoom.us/docs/internal-apps/s2s-oauth/
- Cloud recording files include a `TRANSCRIPT` file type in VTT format
  with a `download_url`, fetched with the same bearer token: confirmed
  across multiple Zoom Developer Forum threads on downloading transcript
  VTT files (searched 2026-09-12; direct `WebFetch` of
  developers.zoom.us's meetings-API reference page returned no usable
  body content on 2026-09-12).
- Meeting participants: `GET /report/meetings/{meetingId}/participants`
  returns `join_time` / `leave_time` / `user_name` and an `email` field
  when the participant joined signed in: Zoom Developer Forum threads on
  the past-meeting participants report (searched 2026-09-12).

Not independently verified against a live Zoom account: the exact list
endpoint used here for "past meetings with recordings"
(`GET /users/{userId}/recordings`, cursor-paginated via
`next_page_token`) is Zoom's long-documented Cloud Recording list
endpoint, but the precise `meetings[].recording_files[]` field set
should be confirmed against a real account before go-live.
"""

from __future__ import annotations

import base64
import hashlib
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlsplit

import httpx

from portolan.connectors.base import (
    SourceRef,
    Transport,
    httpx_transport,
    iter_date_windows,
    parse_datetime,
    request_with_reauth,
    request_with_retry,
    require_allowed_origin,
    source_day,
)
from portolan.extract.schema import SourceType
from portolan.ingest.models import Participant, RawSource
from portolan.util import clock

__all__ = ["ZoomConfig", "ZoomConnector", "convert_vtt_to_call_lines"]

_VTT_TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[.,]\d{3}\s*-->")
_SPEAKER_PREFIX_RE = re.compile(r"^[^:\n]{1,60}:\s")


def _zoom_source_id(meeting: dict) -> str:
    """A hash of the meeting UUID. The UUID is base64 and can contain `/`
    and `+`, which do not belong in a source slug or a URL path segment.
    Listing and `fetch` both use this, so their `source_id`s agree.
    """
    raw = str(meeting.get("uuid") or meeting.get("id"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ZoomConfig:
    """Server-to-Server OAuth app credentials, never from code."""

    account_id: str
    client_id: str
    client_secret: str
    base_url: str = "https://api.zoom.us/v2"
    oauth_url: str = "https://zoom.us/oauth/token"
    user_id: str = "me"


def convert_vtt_to_call_lines(vtt_text: str) -> list[str]:
    """Convert a WebVTT transcript into `[HH:MM:SS] Speaker: text` lines
    matching `corpus/out/calls/*.txt` (one line per cue, timestamp
    truncated to whole seconds). Zoom's own transcript VTT already
    prefixes each cue's text with `Speaker Name: `; a cue lacking that
    prefix falls back to `Unknown Speaker: `."""
    lines: list[str] = []
    for block in re.split(r"\n\s*\n", vtt_text.strip()):
        block_lines = [line for line in block.splitlines() if line.strip()]
        if not block_lines or block_lines[0].strip().upper() == "WEBVTT":
            continue

        timing_index = next(
            (i for i, line in enumerate(block_lines) if _VTT_TIME_RE.match(line.strip())),
            None,
        )
        if timing_index is None:
            continue

        match = _VTT_TIME_RE.match(block_lines[timing_index].strip())
        assert match is not None
        hours, minutes, seconds = match.groups()
        timestamp = f"{hours}:{minutes}:{seconds}"

        text = " ".join(line.strip() for line in block_lines[timing_index + 1 :] if line.strip())
        if not text:
            continue
        if _SPEAKER_PREFIX_RE.match(text):
            lines.append(f"[{timestamp}] {text}")
        else:
            lines.append(f"[{timestamp}] Unknown Speaker: {text}")
    return lines


# Zoom's recordings list accepts a `from`/`to` span of about a month per
# request, so a longer backfill is split into windows.
_ZOOM_MAX_WINDOW_DAYS = 30


class ZoomConnector:
    """`build-contracts.md`'s `Connector` protocol over the Zoom API.
    `list_sources` lists recordings and keeps only meetings with a VTT
    transcript file; `fetch` downloads that transcript and the
    participants report.

    When `since` is given, `list_sources` chunks `[since, today]` into
    `_ZOOM_MAX_WINDOW_DAYS`-day windows so a wide range is not cut to
    Zoom's first window. `since=None` sends no `from`/`to`, so Zoom applies
    its own default window.
    """

    name = "zoom"

    def __init__(
        self,
        config: ZoomConfig,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or httpx_transport(httpx.Client(timeout=30.0))
        self._sleep = sleep
        self._access_token: str | None = None

    def _authenticate(self) -> None:
        if self._access_token is not None:
            return
        basic = base64.b64encode(
            f"{self._config.client_id}:{self._config.client_secret}".encode()
        ).decode()
        response = request_with_retry(
            self._transport,
            "POST",
            self._config.oauth_url,
            params={"grant_type": "account_credentials", "account_id": self._config.account_id},
            headers={"Authorization": f"Basic {basic}"},
            sleep=self._sleep,
            vendor="zoom",
        )
        self._access_token = response.json()["access_token"]

    def _headers(self) -> dict[str, str]:
        self._authenticate()
        return {"Authorization": f"Bearer {self._access_token}"}

    def _force_reauth_headers(self) -> dict[str, str]:
        """Discard the cached token and authenticate again: the
        `reauthenticate` callback for `request_with_reauth`."""
        self._access_token = None
        return self._headers()

    def list_sources(self, since: date | None) -> Iterable[SourceRef]:
        if since is None:
            yield from self._list_recordings_window(from_date=None, to_date=None)
            return
        for window_from, window_to in iter_date_windows(
            since, clock.today(), _ZOOM_MAX_WINDOW_DAYS
        ):
            yield from self._list_recordings_window(from_date=window_from, to_date=window_to)

    def _list_recordings_window(
        self, *, from_date: date | None, to_date: date | None
    ) -> Iterable[SourceRef]:
        params: dict[str, object] = {"page_size": 30}
        if from_date is not None:
            params["from"] = from_date.isoformat()
        if to_date is not None:
            params["to"] = to_date.isoformat()
        while True:
            response = request_with_reauth(
                self._transport,
                "GET",
                f"{self._config.base_url}/users/{self._config.user_id}/recordings",
                params=params,
                headers=self._headers(),
                reauthenticate=self._force_reauth_headers,
                sleep=self._sleep,
                vendor="zoom",
            )
            payload = response.json()
            for meeting in payload.get("meetings", []):
                transcript_file = next(
                    (
                        f
                        for f in meeting.get("recording_files", [])
                        if f.get("file_type") == "TRANSCRIPT"
                    ),
                    None,
                )
                if transcript_file is None:
                    continue
                yield SourceRef(
                    source_id=f"zoom-{_zoom_source_id(meeting)}",
                    source_type=SourceType.CALL,
                    updated_at=parse_datetime(meeting["start_time"]),
                    ref={"meeting": meeting, "transcript_file": transcript_file},
                )
            token = payload.get("next_page_token")
            if not token:
                break
            params = dict(params, next_page_token=token)

    def _list_participants(self, meeting_id: object) -> list[Participant]:
        participants: list[Participant] = []
        params: dict[str, object] = {"page_size": 30}
        while True:
            response = request_with_reauth(
                self._transport,
                "GET",
                f"{self._config.base_url}/report/meetings/{meeting_id}/participants",
                params=params,
                headers=self._headers(),
                reauthenticate=self._force_reauth_headers,
                sleep=self._sleep,
                vendor="zoom",
            )
            payload = response.json()
            for p in payload.get("participants", []):
                participants.append(
                    Participant(
                        name=p.get("name") or p.get("user_name") or "Unknown", email=p.get("email")
                    )
                )
            token = payload.get("next_page_token")
            if not token:
                break
            params = dict(params, next_page_token=token)
        return participants

    def fetch(self, ref: SourceRef) -> RawSource:
        meeting = ref.ref["meeting"]
        transcript_file = ref.ref["transcript_file"]

        download_url = transcript_file["download_url"]
        # The bearer token goes with this request. Zoom's examples serve
        # transcripts from the OAuth host (zoom.us), not the REST host, so
        # both configured hosts are allowed.
        require_allowed_origin(
            download_url,
            allowed_hosts=[
                urlsplit(self._config.base_url).hostname or "",
                urlsplit(self._config.oauth_url).hostname or "",
            ],
            vendor="zoom",
        )
        vtt_response = request_with_reauth(
            self._transport,
            "GET",
            download_url,
            headers=self._headers(),
            reauthenticate=self._force_reauth_headers,
            sleep=self._sleep,
            vendor="zoom",
        )
        vtt_text = vtt_response.text
        text = "\n".join(convert_vtt_to_call_lines(vtt_text))

        meeting_id = meeting.get("id")
        participants = self._list_participants(meeting_id)

        return RawSource(
            source_id=f"zoom-{_zoom_source_id(meeting)}",
            source_type=SourceType.CALL,
            source_date=source_day(parse_datetime(meeting["start_time"])),
            participants=participants,
            account_hints=[],
            text=text,
            original_bytes=vtt_text.encode("utf-8"),
            original_content_type="text/vtt",
            metadata={"meeting_id": meeting_id, "topic": meeting.get("topic")},
        )
