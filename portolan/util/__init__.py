"""Small, dependency-free helpers shared across `portolan` modules that
don't belong to any one wave's package: `portolan.util.clock` (the
e2e-run follow-up wave's item 2, "One clock") and `portolan.util.redact`
(`redact_dsn`, moved here from `portolan.web.demo` by the 2026-09-18 ops
bug hunt so `portolan.db` -- a lower layer than `portolan.web` -- can
redact a DSN too).
"""

from __future__ import annotations
