"""Does a quote appear on a page? The production quote gate's rule
(`portolan.extract.quote_gate.quote_is_supported`), for a claim's quote on
its original, and the row-by-field variant for a CRM row. Shared by
`scripts/verify.py` and the `portolan` CLI."""

from __future__ import annotations

import re

from portolan.client.htmltree import Node

# `portolan.extract.quote_gate.FUZZY_QUOTE_THRESHOLD`, pinned by
# `tests/test_verify_script.py`.
FUZZY_QUOTE_THRESHOLD = 0.75
_GLYPH_FOLD = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."}
)
_QUOTE_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_GLYPH_FOLD).strip().lower())


def quote_match(quote: str, text: str) -> str:
    """`"exact"`, `"fuzzy"` or `"none"`: whether `quote` is on `text` by the
    production quote gate's rule (`portolan.extract.quote_gate.
    quote_is_supported`): glyph-folded containment, else an in-order token
    match of at least `FUZZY_QUOTE_THRESHOLD`."""
    if not quote.strip() or not text.strip():
        return "none"
    if fold(quote) in fold(text):
        return "exact"
    quote_tokens = _QUOTE_TOKEN_RE.findall(quote.translate(_GLYPH_FOLD).lower())
    text_tokens = _QUOTE_TOKEN_RE.findall(text.translate(_GLYPH_FOLD).lower())
    if not quote_tokens:
        return "none"
    pos = matched = 0
    for token in quote_tokens:
        try:
            idx = text_tokens.index(token, pos)
        except ValueError:
            continue
        matched += 1
        pos = idx + 1
    return "fuzzy" if matched / len(quote_tokens) >= FUZZY_QUOTE_THRESHOLD else "none"


def row_quote_match(quote: str, fields: dict[str, str], label: str = "") -> str:
    """`"fields"` or `"none"`: whether `quote` is made of one CRM row's
    fields. Each `key=value` or `key: value` it states (the forms the file
    and API connectors render a row in for extraction) must be a field with
    that value; any other words must be the row's object `label` or part of
    a field's value. The viewer lists a row's fields in stored (key) order,
    not the export's column order the extractor read, so a quote spanning
    fields is on the page field by field, never as one string."""
    text = fold(quote)
    if not text:
        return "none"
    covered = [False] * len(text)
    for key, value in fields.items():
        value = fold(value)
        if not value:
            continue
        for form in (f"{fold(key)}={value}", f"{fold(key)}: {value}"):
            start = text.find(form)
            while start != -1:
                end = start + len(form)
                if end == len(text) or text[end] in ",; ":
                    covered[start:end] = [True] * len(form)
                start = text.find(form, start + 1)
    rest = "".join(" " if covered[i] else ch for i, ch in enumerate(text))
    values = [fold(v) for v in fields.values() if v.strip()]
    label = fold(label)
    found = any(covered)
    for piece in (p.strip() for p in re.split(r"[,;]", rest)):
        if label and (piece == label or piece.startswith(f"{label}:")):
            piece = piece[len(label) :].lstrip(":").strip()
        if not piece:
            continue
        if not any(piece in v for v in values):
            return "none"
        found = True
    return "fields" if found else "none"


def row_fields(block: Node) -> tuple[dict[str, str], str]:
    """A CRM row block's fields (`div.crm-field[data-field]`, value in
    `span.crm-value`) and its object label; `({}, "")` for any other block."""
    fields = {}
    for field in block.find_all("div", "crm-field"):
        value = field.find("span", "crm-value")
        fields[field.attrs.get("data-field", "")] = value.text() if value is not None else ""
    label = block.find("p", "crm-object")
    return fields, label.text() if label is not None else ""
