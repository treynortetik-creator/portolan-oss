"""Ingest: intake, prepare, filter, ledger (portolan-spec-v2.md sec 5.1).

    intake -> filter -> extract -> resolve -> merge -> write -> index

This package owns the first two steps (`intake`, `prepare_text`,
`customer_filter`) plus the coverage ledger. Extraction onward lives in
other packages.
"""

from __future__ import annotations
