"""Tests for `portolan.util.redact.redact_dsn`, moved out of
`portolan.web.demo` (bug hunt 2026-09-18, ops review) so `portolan.db`'s
new backup/restore CLI can redact a DSN without importing `portolan.web`
from a lower layer. `portolan.web.demo` still re-exports the same name,
covered here too so nothing that imports it from there breaks.
"""

from __future__ import annotations

from portolan.util.redact import redact_dsn
from portolan.web.demo import redact_dsn as demo_redact_dsn


def test_redacts_the_password_between_scheme_and_host() -> None:
    dsn = "postgresql://portolan:super-secret@db.example.com:5432/portolan"
    redacted = redact_dsn(dsn)
    assert "super-secret" not in redacted
    assert redacted == "postgresql://portolan:***@db.example.com:5432/portolan"


def test_leaves_a_passwordless_dsn_unchanged() -> None:
    dsn = "postgresql://localhost/portolan_test"
    assert redact_dsn(dsn) == dsn


def test_web_demo_re_exports_the_same_function() -> None:
    assert demo_redact_dsn is redact_dsn
