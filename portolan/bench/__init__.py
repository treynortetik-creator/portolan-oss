"""Portolan bench: score extraction and the graph against the synthetic
corpus's answer key (`corpus/truth-table.yaml` + `corpus/out/manifest.json`).

See `01-spec/quality-loop.md` ("The bench") and `01-spec/build-contracts.md`
("Bench") for the design this module implements.
"""

from __future__ import annotations
