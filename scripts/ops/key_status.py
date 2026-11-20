"""Print the OpenRouter key's spend limits without ever printing the key.

    uv run python scripts/ops/key_status.py

Reads `OPENROUTER_API_KEY` in-process through `portolan.config.get_settings()` (the
environment first, then the repo's `.env`), calls `GET https://openrouter.ai/api/v1/key`, and
prints four numbers: `limit`, `limit_remaining`, `usage`, `usage_daily`. Nothing else from the
response is printed, because the response also carries a `label` that shows part of the key.

Why this exists (guardrails plan 2026-09-22, rank 2): `.claude/hooks/no-secrets.py` blocks
reading `.env` from the shell, which was how the orchestrator checked the key's balance. This
is the replacement. The call is free (no model is invoked), so it has no spend line.

Exit codes: 0 printed the numbers, 2 no key configured, 1 the request failed.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KEY_URL = "https://openrouter.ai/api/v1/key"
FIELDS = ("limit", "limit_remaining", "usage", "usage_daily")


def read_key() -> str | None:
    from portolan.config import get_settings, secret_value

    key = secret_value(get_settings().openrouter_api_key)
    return key.strip() if key and key.strip() else None


def fetch_status(key: str, opener: Callable = urllib.request.urlopen) -> dict:
    request = urllib.request.Request(
        KEY_URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}
    )
    with opener(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    return {name: data.get(name) for name in FIELDS}


def _scrub(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def main(
    argv: list[str] | None = None,
    *,
    key_reader: Callable[[], str | None] = read_key,
    opener: Callable = urllib.request.urlopen,
) -> int:
    key = key_reader()
    if not key:
        print("OPENROUTER_API_KEY is not set in the environment or .env", file=sys.stderr)
        return 2
    try:
        status = fetch_status(key, opener)
    except urllib.error.HTTPError as exc:
        # The status line only: an error body is not guaranteed to be free of the key.
        print(f"GET {KEY_URL} failed: HTTP {exc.code}", file=sys.stderr)
        return 1
    except Exception as exc:  # network, JSON -- never let the key ride out in a traceback
        print(
            f"GET {KEY_URL} failed: {_scrub(type(exc).__name__ + ': ' + str(exc), key)}",
            file=sys.stderr,
        )
        return 1
    for name in FIELDS:
        print(f"{name}: {status[name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
