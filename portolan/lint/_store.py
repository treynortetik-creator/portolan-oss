"""The `Store` type the `missing_original` check depends on, re-exported from
`portolan.store` so `portolan.lint` imports it from one place.
"""

from __future__ import annotations

from portolan.store import Store

__all__ = ["Store"]
