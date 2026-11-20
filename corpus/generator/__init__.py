"""Synthetic corpus generator for Portolan.

Turns `corpus/truth-table.yaml` (the answer key) into a corpus of realistic
customer-facing sources under `corpus/out/`, plus a `manifest.json` that
records exactly where every planted fact landed. The extraction bench
(`01-spec/quality-loop.md`) scores a model's output against that manifest.

Modules:
    briefs      truth table -> one Brief per source (facts, positions, phrasing)
    structured  CRM CSVs, RSS JSON, email headers, ticket skeletons, locator formats
    prose       brief -> source text, via API or manual (hand-authored) mode
    markers     find/verify/strip ``⟦F01⟧ ... ⟦/F01⟧`` fact markers, record locators
    manifest    manifest schema, writer, and validator against the truth table
    cli         ``python -m corpus.generator.cli``
"""

from __future__ import annotations

__version__ = "0.1.0"
