"""The CLI's shared parts: exit codes, the session file, and `Ctx`, which
turns the global options into a target, keys and one signed-in client."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from portolan.client.http import PortolanClient
from portolan.client.keys import (
    DEFAULT_MCP_KEY_FILE,
    DEFAULT_ORG_KEY_FILE,
    MCP_KEY_ENV,
    ORG_KEY_ENV,
    Key,
    resolve_key,
)

EXIT_OK = 0
EXIT_FAILED = 1  # an assertion or validation failed
EXIT_USAGE = 2
EXIT_REFUSED = 3  # the app said no: a 4xx or an auth failure
EXIT_ERROR = 4  # the app errored or timed out

SESSION_ENV = "PORTOLAN_SESSION"
BASE_URL_ENV = "PORTOLAN_BASE_URL"
DEFAULT_SESSION_FILE = ".portolan-session.json"
DEFAULT_LOCAL_URL = "http://127.0.0.1:8000"
DEFAULT_EDITOR = "portolan-cli"
LOG_LIMIT = 200
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class CliError(Exception):
    """A failure with an exit code. `message` is the whole explanation."""

    def __init__(self, message: str, code: int = EXIT_FAILED) -> None:
        super().__init__(message)
        self.code = code


class UsageError(CliError):
    def __init__(self, message: str) -> None:
        super().__init__(message, EXIT_USAGE)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Session:
    """`.portolan-session.json`: where the last command left off. Holds the
    target, the chat conversation and a log of commands, never a key or a
    cookie: every call signs in again."""

    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def path_for(cls, flag: str | None) -> Path:
        return Path(flag or os.environ.get(SESSION_ENV) or DEFAULT_SESSION_FILE)

    @classmethod
    def load(cls, path: Path) -> Session | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CliError(f"cannot read the session file {path}: {exc}", EXIT_FAILED) from exc
        if not isinstance(data, dict) or data.get("version") != 1:
            raise CliError(f"{path} is not a portolan session file (version 1)", EXIT_FAILED)
        return cls(path, data)

    @classmethod
    def create(cls, path: Path, *, target: str, base_url: str | None, editor: str) -> Session:
        session = cls(
            path,
            {
                "version": 1,
                "created": _now(),
                "target": target,
                "base_url": base_url,
                "editor": editor,
                "conversation_id": None,
                "log": [],
            },
        )
        session.save()
        return session

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set_conversation(self, conversation_id: int | None) -> None:
        self.data["conversation_id"] = conversation_id
        self.save()

    def log(self, command: str, exit_code: int) -> None:
        entries = self.data.setdefault("log", [])
        entries.append({"at": _now(), "command": command, "exit": exit_code})
        del entries[:-LOG_LIMIT]
        self.save()


GLOBAL_OPTIONS = (
    "json",
    "session",
    "target",
    "base_url",
    "editor",
    "org_key_file",
    "mcp_key_file",
    "timeout",
    "chat_timeout",
)


class Ctx:
    """Global options, the session (if there is one) and a lazily made,
    signed-in client shared by every command in this process."""

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options
        self.json: bool = bool(options.get("json"))
        self.session_path = Session.path_for(options.get("session"))
        self.session = Session.load(self.session_path)
        self._client: PortolanClient | None = None

    def option(self, name: str, default: Any = None) -> Any:
        value = self.options.get(name)
        return default if value is None else value

    @property
    def target(self) -> str:
        return (
            self.option("target")
            or (self.session.get("target") if self.session else None)
            or "local"
        )

    @property
    def base_url(self) -> str:
        target = self.target
        if target == "local":
            url = (
                self.option("base_url")
                or (self.session.get("base_url") if self.session else None)
                or os.environ.get(BASE_URL_ENV)
                or DEFAULT_LOCAL_URL
            )
            host = urlsplit(url).hostname or ""
            if host not in _LOOPBACK_HOSTS:
                raise UsageError(
                    f"--target local only talks to a local app, not {host or url!r}; "
                    "pass --target <base-url> to reach another server"
                )
            return url.rstrip("/")
        if self.option("base_url"):
            raise UsageError(
                "--base-url goes with --target local; a URL target is its own base URL"
            )
        parts = urlsplit(target)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise UsageError(f"unknown target {target!r}: use 'local' or a URL like https://host")
        return target.rstrip("/")

    @property
    def editor(self) -> str:
        return (
            self.option("editor")
            or (self.session.get("editor") if self.session else None)
            or DEFAULT_EDITOR
        )

    def keys(self) -> tuple[Key, Key]:
        local = self.target == "local"
        org = resolve_key(
            _path_or_none(self.option("org_key_file")),
            DEFAULT_ORG_KEY_FILE,
            ORG_KEY_ENV,
            prefer_env=local,
        )
        mcp = resolve_key(
            _path_or_none(self.option("mcp_key_file")),
            DEFAULT_MCP_KEY_FILE,
            MCP_KEY_ENV,
            prefer_env=local,
        )
        return org, mcp

    @property
    def client(self) -> PortolanClient:
        if self._client is None:
            org, mcp = self.keys()
            kwargs: dict[str, Any] = {}
            if self.option("timeout"):
                kwargs["timeout"] = float(self.option("timeout"))
            if self.option("chat_timeout"):
                kwargs["chat_timeout"] = float(self.option("chat_timeout"))
            self._client = PortolanClient(
                self.base_url, org_key=org, mcp_key=mcp, editor=self.editor, **kwargs
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def _path_or_none(value: Any) -> Path | None:
    return Path(value) if value else None
