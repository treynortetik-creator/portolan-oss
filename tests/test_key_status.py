"""`scripts/ops/key_status.py` prints the OpenRouter key's four spend numbers and never the
key (guardrails plan 2026-09-22, rank 2). No network: the HTTP opener is a fake.

`scripts/` is excluded from the container image, so this module skips there (the pattern
`tests/test_silver_label_defaults.py` uses)."""

from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from types import ModuleType

from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "ops", "key_status.py")

FAKE_KEY = "fake-openrouter-key-for-tests-0123456789abcdef"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("key_status_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _opener_returning(payload: dict, seen: list):
    def opener(request, timeout):
        seen.append(request)
        return _Response(json.dumps(payload).encode("utf-8"))

    return opener


def test_prints_the_four_numbers_and_nothing_else(capsys) -> None:
    ks = _load()
    seen: list = []
    payload = {
        "data": {
            "label": FAKE_KEY[:12] + "...",
            "limit": 50,
            "limit_remaining": 12.5,
            "usage": 37.5,
            "usage_daily": 1.25,
            "is_free_tier": False,
        }
    }
    code = ks.main([], key_reader=lambda: FAKE_KEY, opener=_opener_returning(payload, seen))
    out, err = capsys.readouterr()
    assert code == 0
    assert out.splitlines() == [
        "limit: 50",
        "limit_remaining: 12.5",
        "usage: 37.5",
        "usage_daily: 1.25",
    ]
    assert err == ""
    assert FAKE_KEY[:12] not in out  # the label (a key prefix) is dropped
    assert seen[0].full_url == "https://openrouter.ai/api/v1/key"
    assert seen[0].get_header("Authorization") == f"Bearer {FAKE_KEY}"


def test_missing_key_exits_2_without_calling_out(capsys) -> None:
    ks = _load()

    def opener(request, timeout):  # pragma: no cover - must not be reached
        raise AssertionError("no key, so no request")

    assert ks.main([], key_reader=lambda: None, opener=opener) == 2
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().err


def test_http_error_prints_the_status_only(capsys) -> None:
    ks = _load()

    def opener(request, timeout):
        raise urllib.error.HTTPError(
            ks.KEY_URL, 401, "Unauthorized", {}, io.BytesIO(f'{{"key": "{FAKE_KEY}"}}'.encode())
        )

    assert ks.main([], key_reader=lambda: FAKE_KEY, opener=opener) == 1
    out, err = capsys.readouterr()
    assert "HTTP 401" in err
    assert FAKE_KEY not in out + err


def test_other_failures_scrub_the_key(capsys) -> None:
    ks = _load()

    def opener(request, timeout):
        raise OSError(f"connection reset while sending Bearer {FAKE_KEY}")

    assert ks.main([], key_reader=lambda: FAKE_KEY, opener=opener) == 1
    out, err = capsys.readouterr()
    assert "connection reset" in err
    assert FAKE_KEY not in out + err


def test_read_key_uses_settings(monkeypatch) -> None:
    ks = _load()
    monkeypatch.setenv("OPENROUTER_API_KEY", f"  {FAKE_KEY}\n")
    assert ks.read_key() == FAKE_KEY
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    assert ks.read_key() is None
