"""Stub writers for the WP6 noise tests. No network.

`rich` is WP4's obedient fake (`tests.v2_writer_fakes.obedient`) with
richer fixture words, so every noise operation has something to act on:
homophones and droppable words, curly apostrophes and a dash (mojibake),
and an ampersand (HTML entities). `neardup_reply` is an obedient
near-duplicate writer: it returns the ORIGINAL from the prompt in its own
protocol, with the CHANGE applied inside the named statement and each
statement's date said the way its WHEN line gives.
"""

from __future__ import annotations

import re
from datetime import datetime

from tests.v2_writer_fakes import obedient

_RICH = {
    "fixture opening words": (
        "Morning all — the weather’s fine here, so we’re just going to start. "
        "It’s their turn to share the screen & the notes."
    ),
    "fixture reply words": (
        "Thanks, the new link works on our side and we can hear you right now, not a problem."
    ),
    "fixture closing words": (
        "Great, we’ll put the next check-in on the calendar. Talk soon, and have a good week."
    ),
    "fixture greeting": "Hi there,",
    "fixture lead ⟦": "Just so you know, ⟦",
    "fixture span number": "the fixture span, it’s their number",
}


def rich(model: str, system: str, user: str, n: int) -> str:
    text = obedient(model, system, user, n)
    if "REWRITE THESE UNITS" in user:
        return text
    for a, b in _RICH.items():
        text = text.replace(a, b)
    return re.sub(r"(⟦/[FD]\d+⟧) tail", r"\1 — so that’s where their team is at.", text)


_CHANGE_RE = re.compile(
    r"^CHANGE: in the statement marked ⟦(\w+)⟧, .*? is now (.+?) instead of (.+?)\.$", re.M
)
_ITEM_RE = re.compile(r"^\d+\. ⟦(\w+)⟧")
_WHEN_RE = re.compile(r'^\s+WHEN: say the date as "([^"]+)"')


def _forms(human: str) -> list[str]:
    try:
        when = datetime.strptime(human, "%B %d, %Y")
    except ValueError:
        return [human]
    return [human, f"{when.strftime('%B')} {when.day}", when.date().isoformat()]


def _whens(user: str) -> dict[str, str | None]:
    """Every marked statement id -> the first date phrase its WHEN line gives."""
    out: dict[str, str | None] = {}
    current = None
    for line in user.split("\n"):
        m = _ITEM_RE.match(line)
        if m:
            current = m.group(1)
            out[current] = None
            continue
        w = _WHEN_RE.match(line)
        if w and current:
            out[current] = w.group(1)
    return out


def neardup_reply(
    user: str,
    *,
    keep_old: bool = False,
    drop_marker: str | None = None,
    add_sentence: str | None = None,
) -> str:
    original = user.split("\nORIGINAL (", 1)[1].split("\n", 1)[1].split("\n\nCHANGE:", 1)[0]
    change = _CHANGE_RE.search(user)
    assert change is not None
    marker, new_h, old_h = change.groups()
    text = original
    for item, when in _whens(user).items():
        span = re.search(rf"⟦{item}⟧(.*?)⟦/{item}⟧", text)
        if span is None:
            continue
        body = span.group(1)
        if item == marker:
            if keep_old:
                continue
            for old, new in zip(_forms(old_h), _forms(new_h), strict=True):
                body = body.replace(old, new)
            old_n, new_n = re.match(r"\d+", old_h), re.match(r"\d+", new_h)
            if old_n and new_n:  # a count or a percentage, said as a bare number
                body = re.sub(rf"(?<!\d){old_n.group(0)}(?!\d)", new_n.group(0), body)
        if when and when not in body:
            body = f"{body} {when}"
        text = text[: span.start(1)] + body + text[span.end(1) :]
    if drop_marker is not None:
        text = re.sub(rf"⟦/?{drop_marker}⟧", "", text)
    if add_sentence is not None:
        text = text.rstrip() + ". " + add_sentence
    return text + "\n"
