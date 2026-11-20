"""Signed, opaque paging cursors for the MCP door (checklist item 26(c),
`01-spec/app-checklist-2026-09-18.md`; idea from `01-spec/research/
openworkgraph-2026-09-22.md` sec 4, reimplemented, nothing copied).

A cursor carries everything the next page needs -- the frozen snapshot
boundary, the keyset position, which listing it belongs to and for which
tenant -- so the server keeps no paging state at all. It is *signed, not
encrypted*: `pc1.<base64url JSON>.<base64url HMAC-SHA256>`. A client can
read it but cannot change it; any edit, truncation, a cursor minted under
a different key, for a different tenant, or for a different listing, is
rejected with `InvalidCursorError` (the door turns that into a tool error
telling the model to start again without a cursor).

The HMAC key is derived (never used raw) from the hosted transport's
`PORTOLAN_MCP_ORG_KEY` when one is configured, so cursors survive a
restart and work across replicas; with no org key (a local stdio server)
it is a random per-process key, and a cursor lives as long as the process
that issued it. Rotating the org key invalidates outstanding cursors,
which is the right failure: the client starts the listing over.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from typing import Any

CURSOR_VERSION = "pc1"
MAX_CURSOR_CHARS = 4_096
_RESERVED = frozenset({"v", "t", "k"})


class InvalidCursorError(ValueError):
    """A cursor that is malformed, tampered with, or not for this listing."""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def derive_cursor_key(secret: str | bytes | None) -> bytes:
    """A 32-byte HMAC key for cursors: derived from `secret` with a fixed
    domain-separation label (so the cursor key is never the org key
    itself), or random when `secret` is `None`/empty."""
    if not secret:
        return secrets.token_bytes(32)
    raw = secret.encode("utf-8") if isinstance(secret, str) else secret
    return hmac.new(raw, b"portolan-mcp-cursor-v1", hashlib.sha256).digest()


class CursorCodec:
    def __init__(self, key: bytes, *, tenant: str) -> None:
        if len(key) < 16:
            raise ValueError("cursor key must be at least 16 bytes")
        self._key = key
        self._tenant = tenant

    def _mac(self, payload: str) -> str:
        return _b64e(
            hmac.new(self._key, f"{CURSOR_VERSION}.{payload}".encode(), hashlib.sha256).digest()
        )

    def encode(self, kind: str, state: dict[str, Any]) -> str:
        clash = _RESERVED & state.keys()
        if clash:
            raise ValueError(f"cursor state may not use reserved keys {sorted(clash)}")
        body = {"v": 1, "t": self._tenant, "k": kind, **state}
        payload = _b64e(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return f"{CURSOR_VERSION}.{payload}.{self._mac(payload)}"

    def decode(self, token: Any, *, kind: str) -> dict[str, Any]:
        """The state `encode` was given, or `InvalidCursorError`."""
        if not isinstance(token, str) or not token:
            raise InvalidCursorError("cursor is empty or not a string")
        if len(token) > MAX_CURSOR_CHARS:
            raise InvalidCursorError("cursor is too long to be one this server issued")
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != CURSOR_VERSION:
            raise InvalidCursorError("cursor is malformed")
        _, payload, mac = parts
        if not hmac.compare_digest(mac.encode("ascii", "replace"), self._mac(payload).encode()):
            raise InvalidCursorError(
                "cursor signature does not match: it was altered, truncated, or issued by a "
                "server with a different key"
            )
        try:
            body = json.loads(_b64d(payload))
        except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
            raise InvalidCursorError("cursor is malformed") from exc
        if not isinstance(body, dict) or body.get("v") != 1:
            raise InvalidCursorError("cursor is from an unsupported version")
        if body.get("t") != self._tenant:
            raise InvalidCursorError("cursor was issued for a different tenant")
        if body.get("k") != kind:
            raise InvalidCursorError(f"cursor belongs to a different listing, not {kind!r}")
        return {key: value for key, value in body.items() if key not in _RESERVED}
