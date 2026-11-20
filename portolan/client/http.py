"""`PortolanClient`: a running Portolan app driven over HTTP the way a person
in a browser drives it (sign in with the org key, then the same pages and
routes). The `portolan` CLI is built on it; `scripts/verify.py` shares the
HTTP client factory, the chat stream and the citation checks."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from portolan.client.chat import classify_outcome, resolve_citation, stream_question
from portolan.client.htmltree import Page
from portolan.client.keys import Key
from portolan.client.mcp import McpSession

# `portolan/web/auth.py` SESSION_COOKIE_NAME.
SESSION_COOKIE = "portolan_session"
DEFAULT_TIMEOUT_SECONDS = 30.0
# The chat stream sends nothing until the model has answered, so its read
# timeout covers a whole model call.
DEFAULT_CHAT_TIMEOUT_SECONDS = 180.0


def make_http_client(timeout: float, user_agent: str) -> httpx.Client:
    return httpx.Client(timeout=timeout, headers={"User-Agent": user_agent})


class AppRefused(Exception):
    """The app said no: a 4xx, a redirect to /login, or a missing or wrong key."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class AppFailed(Exception):
    """The app errored (5xx) or could not be reached in time."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _http_error(exc: httpx.HTTPError) -> AppFailed:
    return AppFailed(f"{type(exc).__name__}: {exc}")


class PortolanClient:
    def __init__(
        self,
        base_url: str,
        *,
        org_key: Key,
        mcp_key: Key | None = None,
        editor: str = "portolan-cli",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        chat_timeout: float = DEFAULT_CHAT_TIMEOUT_SECONDS,
        user_agent: str = "portolan-cli/1",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.org_key = org_key
        self.mcp_key = mcp_key or Key()
        self.editor = editor
        self.timeout = timeout
        self.chat_timeout = chat_timeout
        self.session = make_http_client(timeout, user_agent)
        self.anon = make_http_client(timeout, user_agent)
        self._signed_in = False
        self._notes: dict[str, Page | None] = {}
        self._originals: dict[str, Page | None] = {}

    def close(self) -> None:
        self.session.close()
        self.anon.close()

    # ---- sign in and plain requests ---------------------------------

    def login(self) -> None:
        if self._signed_in:
            return
        if not self.org_key.value:
            raise AppRefused(f"no org key: {self.org_key.source}")
        try:
            response = self.session.post(
                f"{self.base_url}/login",
                data={"org_key": self.org_key.value, "editor": self.editor, "next": "/"},
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise _http_error(exc) from exc
        if response.status_code >= 500:
            raise AppFailed(f"login answered {response.status_code}", status=response.status_code)
        if response.status_code != 303 or not response.headers.get("set-cookie"):
            raise AppRefused(
                f"sign-in refused (status {response.status_code}); "
                f"the org key came from {self.org_key.source}",
                status=response.status_code,
            )
        self._signed_in = True

    def _judge(self, response: httpx.Response, expect: tuple[int, ...], path: str) -> None:
        status = response.status_code
        if status in expect:
            return
        if status in (301, 302, 303, 307) and response.headers.get("location", "").startswith(
            "/login"
        ):
            hint = ""
            if urlsplit(self.base_url).scheme == "http":
                hint = (
                    " (a Secure session cookie is not sent over http; start a local app "
                    "with PORTOLAN_COOKIE_SECURE=false)"
                )
            raise AppRefused(f"not signed in: {path} redirected to /login{hint}", status=status)
        if status >= 500:
            raise AppFailed(f"{path} answered {status}", status=status)
        raise AppRefused(f"{path} answered {status}", status=status)

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        anonymous: bool = False,
        expect: tuple[int, ...] = (200,),
    ) -> Page:
        if not anonymous:
            self.login()
        client = self.anon if anonymous else self.session
        try:
            response = client.get(f"{self.base_url}{path}", params=params, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise _http_error(exc) from exc
        self._judge(response, expect, path)
        return Page(response)

    def post(
        self, path: str, data: dict[str, Any] | None = None, *, expect: tuple[int, ...] = (303,)
    ) -> httpx.Response:
        self.login()
        try:
            response = self.session.post(
                f"{self.base_url}{path}", data=data or {}, follow_redirects=False
            )
        except httpx.HTTPError as exc:
            raise _http_error(exc) from exc
        self._judge(response, expect, path)
        return response

    # ---- chat ----------------------------------------------------------

    def _cited(self, cache: dict[str, Page | None], path: str) -> Page | None:
        if path not in cache:
            try:
                cache[path] = self.get(path)
            except (AppRefused, AppFailed):
                cache[path] = None
        return cache[path]

    def ask(self, question: str, conversation: str | int = "new") -> dict[str, Any]:
        """Ask through the browser's own chat stream and follow every
        citation the way a reader clicks it. Never raises for a bad answer:
        the outcome says what came back."""
        self.login()
        streamed = stream_question(
            self.session,
            self.base_url,
            question,
            str(conversation),
            timeout=self.timeout,
            chat_timeout=self.chat_timeout,
        )
        status = streamed["status"]
        if status is not None and status >= 500:
            raise AppFailed(f"chat stream answered {status}", status=status)
        if status is not None and status != 200:
            raise AppRefused(f"chat stream answered {status}: {streamed['error']}", status=status)
        started = time.perf_counter()
        citations = [dict(c) for s in streamed["sentences"] for c in s["citations"]]
        for citation in citations:
            resolve_citation(
                citation,
                lambda path: self._cited(self._notes, path),
                lambda path: self._cited(self._originals, path),
            )
        done = streamed["done"] or {}
        return {
            "question": question,
            "outcome": classify_outcome(streamed),
            "answer": " ".join(s["text"] for s in streamed["sentences"]),
            "sentences": len(streamed["sentences"]),
            "citations": citations,
            "withheld": int(done.get("numerals_dropped") or 0)
            + int(done.get("uncited_dropped") or 0),
            "model": done.get("model"),
            "conversation_id": done.get("conversation_id"),
            "message_id": done.get("message_id"),
            "first_sentence_ms": streamed["first_sentence_ms"],
            "latency_ms": streamed["answer_ms"] or streamed["elapsed_ms"],
            "resolve_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": streamed["error"],
        }

    # ---- MCP -------------------------------------------------------------

    def mcp(self) -> McpSession:
        if not self.mcp_key.value:
            raise AppRefused(f"no MCP key: {self.mcp_key.source}")
        return McpSession(self.base_url, self.mcp_key.value, timeout=self.timeout)
