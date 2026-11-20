"""The hosted MCP door as a client: streamable-HTTP JSON-RPC with the MCP
key in an `Authorization` header. Shared by `scripts/verify.py` (its own
step-by-step checks) and the `portolan` CLI (`McpSession`)."""

from __future__ import annotations

import json
from typing import Any

import httpx

# The protocol revision `tests/test_mcp_server.py` already drives the
# hosted mount with.
MCP_PROTOCOL_VERSION = "2025-06-18"
# The JSON-RPC error a closed AI-access door returns
# (`portolan.mcp_server.access.AI_ACCESS_CLOSED_CODE`, pinned by
# `tests/test_verify_script.py`).
AI_ACCESS_CLOSED_CODE = -32010
ACCEPT = {"Accept": "application/json, text/event-stream"}


class McpError(Exception):
    """The door answered with a JSON-RPC error or an HTTP failure."""

    def __init__(self, message: str, *, code: int | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


def jsonrpc_messages(response: httpx.Response) -> list[dict[str, Any]]:
    """The JSON-RPC messages in a streamable-HTTP response: a JSON body, or
    the `data:` lines of a `text/event-stream` body."""
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        body = response.json()
        return body if isinstance(body, list) else [body]
    messages = []
    for event in response.text.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(
            line[5:].lstrip() for line in event.split("\n") if line.startswith("data:")
        )
        if data:
            messages.append(json.loads(data))
    return messages


def jsonrpc_error_code(response: httpx.Response, request_id: int) -> int | None:
    try:
        for message in jsonrpc_messages(response):
            if message.get("id") == request_id and isinstance(message.get("error"), dict):
                return message["error"].get("code")
    except (ValueError, json.JSONDecodeError):
        return None
    return None


def jsonrpc_result(response: httpx.Response, request_id: int) -> dict[str, Any]:
    for message in jsonrpc_messages(response):
        if message.get("id") == request_id:
            if "error" in message:
                raise ValueError(f"JSON-RPC error {message['error'].get('code')}")
            return message["result"]
    raise ValueError(f"no JSON-RPC response with id {request_id}")


class McpSession:
    """One initialized MCP session: `initialize`, the `initialized`
    notification, then requests, and `close` sends the DELETE."""

    def __init__(self, base_url: str, key: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._auth = {**ACCEPT, "Authorization": f"Bearer {key}"}
        self._http = httpx.Client(timeout=timeout, headers={"User-Agent": "portolan-cli/1"})
        self._headers: dict[str, str] = {}
        self._next_id = 1

    def __enter__(self) -> McpSession:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        try:
            response = self._http.post(f"{self.base_url}/mcp", headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise McpError(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise McpError(
                f"MCP endpoint answered {response.status_code}", status=response.status_code
            )
        return response

    def _request(self, method: str, params: dict[str, Any] | None, headers: dict[str, str]):
        request_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        response = self._post(payload, headers)
        try:
            return response, jsonrpc_result(response, request_id)
        except ValueError as exc:
            code = jsonrpc_error_code(response, request_id)
            raise McpError(str(exc), code=code) from exc
        except json.JSONDecodeError as exc:
            raise McpError(f"unreadable MCP reply: {exc}") from exc

    def open(self) -> None:
        response, result = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "portolan-cli", "version": "1"},
            },
            self._auth,
        )
        session_id = response.headers.get("mcp-session-id")
        if not session_id:
            raise McpError("initialize returned no mcp-session-id")
        self._headers = {
            **self._auth,
            "mcp-session-id": session_id,
            "mcp-protocol-version": result.get("protocolVersion") or MCP_PROTOCOL_VERSION,
        }
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, self._headers)

    def list_tools(self) -> list[dict[str, Any]]:
        return self._request("tools/list", None, self._headers)[1]["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._request("tools/call", {"name": name, "arguments": arguments}, self._headers)[1]

    def close(self) -> None:
        try:
            if self._headers:
                self._http.delete(f"{self.base_url}/mcp", headers=self._headers)
        except httpx.HTTPError:
            pass
        finally:
            self._http.close()
