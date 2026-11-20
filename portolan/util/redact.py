"""`redact_dsn`: strip the password out of a Postgres connection string
before it can reach a log line, an exception message, or stdout.

Moved here (bug hunt 2026-09-18, ops review) from `portolan.web.demo`,
where it started as a one-off helper for that module's own `print()`
call. `portolan.db` needs the same redaction for the new pre-migration
backup / restore CLI (`python -m portolan.db`) -- `portolan.db` is a
lower layer than `portolan.web`, so importing `portolan.web.demo` from
there would invert the dependency direction; `portolan.util` is the
project's existing home for dependency-free helpers shared across
modules (see `portolan.util.clock`). `portolan.web.demo` re-exports this
name so nothing that already imports `redact_dsn` from there breaks.
"""

from __future__ import annotations

import re

__all__ = ["redact_dsn"]

_PASSWORD_RE = re.compile(r"://([^:/@]+):[^@]*@")


def redact_dsn(dsn: str) -> str:
    """Return `dsn` with any password replaced by `***`, safe to log."""
    return _PASSWORD_RE.sub(r"://\1:***@", dsn)
