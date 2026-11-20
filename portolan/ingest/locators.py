"""The regex for parsing a locator marker back out of prepared text.

`portolan.ingest.prepare` marks each locator by wrapping it in square
brackets at the start of a line or paragraph (`[00:14:32]`,
`[S-ticket-027#1]`, `[S-email-005@halvardmedical.com#p2]`,
`[crm/opportunities.csv#O-3001]`, `[url:... 2026-08-26]`). Parse markers
with this module rather than a second pattern.

Only the bracket wrapper lives here; the string inside differs per source
type (`01-spec/provenance-chain.md`'s locator table).
"""

from __future__ import annotations

import re

# `[<anything but brackets>]` at the start of a line, after optional whitespace.
LOCATOR_RE = re.compile(r"^\s*\[([^\[\]]+)\]")

__all__ = ["LOCATOR_RE", "parse_locator"]


def parse_locator(line: str) -> str | None:
    """The locator string inside `line`'s leading `[...]` marker, or
    `None` if the line doesn't start with one."""
    match = LOCATOR_RE.match(line)
    return match.group(1) if match else None
