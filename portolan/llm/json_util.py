"""Pure helpers for pulling a JSON object out of a model's free-text reply
and checking it against a schema. No network and no SDK imports, so they
unit-test without a model call.
"""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

# A fenced code block, ```json ... ``` or plain ``` ... ```. Models often wrap
# JSON in one even when told not to, so fences are tried before the raw text.
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _find_balanced_object(text: str) -> str | None:
    """The first balanced `{...}` span in `text`, or None. Tracks string
    literals so a brace inside a quoted string doesn't unbalance the count
    (a regex cannot handle arbitrary nesting)."""
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        # Unbalanced from this start (a stray "{" in prose): try the next "{".
        search_from = start + 1


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from `text`.

    Tries every fenced code block first (in order), then the raw text
    itself, and within each candidate looks for the first balanced
    `{...}` span. Returns None -- never raises -- if nothing in `text`
    parses as a JSON *object* (a bare list or scalar doesn't count: the
    extraction contract is always an object).
    """
    candidates: list[str] = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    candidates.append(text)

    for candidate in candidates:
        obj_text = _find_balanced_object(candidate)
        if obj_text is None:
            continue
        try:
            parsed = json.loads(obj_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def validate_json_object(obj: dict[str, Any], json_schema: dict[str, Any]) -> str | None:
    """Validate `obj` against `json_schema`. Returns None on success, or the
    validation error message on failure, so callers can build a retry prompt
    without a try/except. Still raises `jsonschema.SchemaError` if
    `json_schema` itself is invalid.
    """
    try:
        jsonschema.validate(obj, json_schema)
    except jsonschema.ValidationError as exc:
        return str(exc)
    return None


def extract_and_validate(
    text: str, json_schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Combine extraction and validation: `(data, None)` on success,
    `(None, error_message)` on either failure mode (nothing extractable,
    or extracted but schema-invalid)."""
    obj = extract_json_object(text)
    if obj is None:
        return None, "no JSON object found in the response"
    error = validate_json_object(obj, json_schema)
    if error is not None:
        return None, error
    return obj, None
