"""Where a client gets the org key and the MCP key. The value never leaves a
`Key` except as a login form field or an `Authorization` header; the only
thing anyone prints is `source`. Shared by `scripts/verify.py` and the
`portolan` CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORG_KEY_FILE = REPO_ROOT / "var" / "railway" / "org-key.txt"
DEFAULT_MCP_KEY_FILE = REPO_ROOT / "var" / "railway" / "mcp-key.txt"
ORG_KEY_ENV = "PORTOLAN_ORG_KEY"
MCP_KEY_ENV = "PORTOLAN_MCP_ORG_KEY"


@dataclass(frozen=True)
class Key:
    value: str | None = None
    source: str = "missing"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_key_file(path: Path) -> Key:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return Key(None, f"missing: cannot read {display_path(path)}")
    if not value:
        return Key(None, f"missing: {display_path(path)} is empty")
    return Key(value, f"file:{display_path(path)}")


def _from_env(env_name: str) -> Key | None:
    value = os.environ.get(env_name, "").strip()
    return Key(value, f"env:{env_name}") if value else None


def resolve_key(
    flag_path: Path | None, default_path: Path, env_name: str, *, prefer_env: bool = False
) -> Key:
    """`--*-key-file` when given; else the default file when it exists; else
    the environment variable. With `prefer_env` the variable comes before
    the default file (a local app has its own key, not the deployed one)."""
    if flag_path is not None:
        return read_key_file(flag_path)
    if prefer_env:
        from_env = _from_env(env_name)
        if from_env is not None:
            return from_env
    if default_path.is_file():
        return read_key_file(default_path)
    from_env = _from_env(env_name)
    if from_env is not None:
        return from_env
    return Key(None, f"missing: no {display_path(default_path)} and {env_name} is unset")
