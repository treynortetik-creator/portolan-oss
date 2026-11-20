"""`expect` conditions for `portolan script`: a metric, an operator and a
value, checked against the result of the step's command.

Three spellings, all the same condition:

    {"outcome": "answered", "citations": ">=2", "cited_notes": "contains signal-"}
    ["citations>=2", "outcome==answered"]
    "citations>=2"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

OPERATORS = (">=", "<=", "==", "!=", ">", "<", "contains")
_OP_RE = re.compile(r"^\s*(>=|<=|==|!=|>|<|contains\b)\s*(.*?)\s*$", re.DOTALL)
_STRING_FORM_RE = re.compile(
    r"^\s*([A-Za-z_][\w.\-]*)\s*(>=|<=|==|!=|>|<|\bcontains\b)\s*(.*?)\s*$", re.DOTALL
)


class ExpectError(ValueError):
    """A condition that cannot be read."""


@dataclass(frozen=True)
class Condition:
    metric: str
    op: str
    want: Any

    def text(self) -> str:
        return f"{self.metric} {self.op} {self.want}"


def _from_pair(metric: str, spec: Any) -> Condition:
    if isinstance(spec, str):
        match = _OP_RE.match(spec)
        if match:
            return Condition(metric, match.group(1), match.group(2))
        return Condition(metric, "==", spec)
    return Condition(metric, "==", spec)


def parse_expect(expect: Any) -> list[Condition]:
    if expect is None:
        return []
    if isinstance(expect, dict):
        return [_from_pair(str(k), v) for k, v in expect.items()]
    items = [expect] if isinstance(expect, str) else expect
    if not isinstance(items, list):
        raise ExpectError(
            f"expect must be an object, a list or a string, not {type(expect).__name__}"
        )
    conditions = []
    for item in items:
        if not isinstance(item, str):
            raise ExpectError(f"a list expect holds strings like 'citations>=2', not {item!r}")
        match = _STRING_FORM_RE.match(item)
        if not match:
            raise ExpectError(f"cannot read expect {item!r}; use '<metric><op><value>'")
        conditions.append(Condition(match.group(1), match.group(2), match.group(3)))
    return conditions


_MISSING = object()


def lookup(data: Any, path: str) -> Any:
    """`a.b.0.c` into dicts and lists; `_MISSING` when any step is absent."""
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.lstrip("-").isdigit():
            try:
                current = current[int(part)]
            except IndexError:
                return _MISSING
        else:
            return _MISSING
    return current


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _equal(got: Any, want: Any) -> bool:
    if isinstance(got, bool):
        return str(want).strip().lower() in (("true", "1", "yes") if got else ("false", "0", "no"))
    if got is None:
        return str(want).strip().lower() in ("none", "null", "")
    if isinstance(got, (list, dict)):
        n = _number(want)
        return n is not None and float(len(got)) == n
    g, w = _number(got), _number(want)
    if g is not None and w is not None:
        return g == w
    return str(got) == str(want)


def _contains(got: Any, want: Any) -> bool:
    needle = str(want).lower()
    if isinstance(got, list):
        return any(needle in str(item).lower() for item in got)
    if isinstance(got, dict):
        return any(needle in str(key).lower() for key in got)
    return needle in str(got).lower()


def compare(got: Any, op: str, want: Any) -> bool:
    if op == "contains":
        return _contains(got, want)
    if op == "==":
        return _equal(got, want)
    if op == "!=":
        return not _equal(got, want)
    left = float(len(got)) if isinstance(got, (list, dict)) else _number(got)
    right = _number(want)
    if left is None or right is None:
        raise ExpectError(f"{op} needs numbers, got {got!r} and {want!r}")
    return {">=": left >= right, "<=": left <= right, ">": left > right, "<": left < right}[op]


def check(conditions: list[Condition], data: Any) -> list[dict[str, Any]]:
    """One record per condition: `ok`, and what the metric actually was."""
    results = []
    for cond in conditions:
        got = lookup(data, cond.metric)
        if got is _MISSING:
            results.append(
                {"expect": cond.text(), "ok": False, "got": None, "error": "no such metric"}
            )
            continue
        try:
            ok = compare(got, cond.op, cond.want)
            error = None
        except ExpectError as exc:
            ok, error = False, str(exc)
        shown = f"{len(got)} items" if isinstance(got, (list, dict)) else got
        results.append({"expect": cond.text(), "ok": ok, "got": shown, "error": error})
    return results
