"""List every change on a branch that could weaken a test or a guardrail.

    uv run python scripts/check_test_edits.py            # base = main, head = HEAD
    uv run python scripts/check_test_edits.py main --head feature/x
    uv run python scripts/check_test_edits.py --worktree # include uncommitted edits

Exit 0 when the branch only ADDS ordinary test lines (new files, new functions, appended
lines); exit 1 when it does any of the following; exit 2 on a git error. The orchestrator runs
this before every merge and reads each hunk it prints (guardrails plan 2026-09-22, rank 6): a
weakened assertion is the cheapest way to turn a red suite green, and a new test cannot catch
it.

- changes or deletes a line of a test file that exists on the base, or deletes or renames one;
- adds a line that switches tests off to a test file that exists on the base, or to any
  `conftest.py`: `skip`, `skipif`, `xfail`, `importorskip`, `skipTest`, `collect_ignore`,
  `pytest_ignore_collect`, `pytest_collection_modifyitems`, or a `return` inside a
  `test_*` function;
- changes a guardrail file in any way, additions included: `.github/workflows/`,
  `.gitleaks.toml`, `.dockerignore`, `docker/`, `.claude/`, `pyproject.toml`,
  `scripts/check_test_edits.py`, `scripts/verify.py`.

Guarded test files: everything under `tests/`, any `test_*.py` / `*_test.py` / `conftest.py`
elsewhere, and the applied migrations with their hash manifest
(`portolan/db/migrations/*.sql`, `MANIFEST.sha256`) -- regenerating the manifest after
editing a migration would otherwise pass `tests/test_migrations_immutable.py`.

The comparison is from the merge-base of base and head, so commits that landed on the base
after the branch point are not counted against the branch.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import PurePosixPath

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# An added line that turns tests off without touching an existing line.
_SWITCH_OFF = re.compile(
    r"\b(?:skip|skipif|xfail|importorskip|skipTest|collect_ignore(?:_glob)?|"
    r"pytest_ignore_collect|pytest_collection_modifyitems)\b"
)
_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_RETURN = re.compile(r"^(\s*)return\b")
_DEF = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)")

GUARDRAIL_FILES = {
    ".gitleaks.toml",
    ".dockerignore",
    "pyproject.toml",
    "scripts/check_test_edits.py",
    "scripts/verify.py",
}
GUARDRAIL_DIRS = (".github/workflows/", "docker/", ".claude/")


class GitError(RuntimeError):
    pass


def git(*args: str) -> str:
    proc = subprocess.run(["git", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def is_guarded(path: str) -> bool:
    p = PurePosixPath(path)
    if p.parts and p.parts[0] == "tests":
        return True
    if fnmatch(p.name, "test_*.py") or fnmatch(p.name, "*_test.py") or p.name == "conftest.py":
        return True
    if p.parent == PurePosixPath("portolan/db/migrations") and (
        p.suffix == ".sql" or p.name == "MANIFEST.sha256"
    ):
        return True
    return False


def is_guardrail(path: str) -> bool:
    return path in GUARDRAIL_FILES or path.startswith(GUARDRAIL_DIRS)


def added_lines(diff_text: str) -> list[tuple[int, str]]:
    """(head line number, text) for every added line in a unified=0 diff."""
    out: list[tuple[int, str]] = []
    new_no = 0
    for line in diff_text.splitlines():
        m = _HUNK.match(line)
        if m:
            new_no = int(m.group(3))
            continue
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            out.append((new_no, line[1:]))
            new_no += 1
    return out


def _enclosing_def(head_lines: list[str], line_no: int, indent: int) -> str | None:
    """Name of the nearest `def` above line_no (1-based) that is indented less than `indent`."""
    for i in range(line_no - 2, -1, -1):
        m = _DEF.match(head_lines[i])
        if m and len(m.group(1)) < indent:
            return m.group(2)
    return None


def switch_off_lines(added: list[tuple[int, str]], head_text: str) -> list[str]:
    """Printable lines for added lines that skip, xfail or deselect tests, or return early
    from a test function."""
    head_lines = head_text.splitlines()
    out = []
    for no, text in added:
        # string contents and comments are data, not code: `_commit_all(repo, "skip it")`
        code = _STRING.sub("''", text).split("#", 1)[0]
        if _SWITCH_OFF.search(code):
            out.append(f"  +{no}: {text}")
            continue
        m = _RETURN.match(code)
        if m:
            name = _enclosing_def(head_lines, no, len(m.group(1)))
            if name and name.startswith("test"):
                out.append(f"  +{no}: {text}")
    return out


@dataclass
class FileReport:
    path: str
    status: str
    lines: list[str] = field(default_factory=list)
    changed: int = 0


def parse_hunks(diff_text: str) -> tuple[list[str], int]:
    """Unified=0 diff -> printable lines for every hunk that removes or replaces lines, and
    the count of removed base lines. Pure-addition hunks are skipped."""
    out: list[str] = []
    removed_total = 0
    old_no = new_no = 0
    in_hunk_with_removal = False
    for line in diff_text.splitlines():
        m = _HUNK.match(line)
        if m:
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            old_no, new_no = int(m.group(1)), int(m.group(3))
            in_hunk_with_removal = old_count > 0
            if in_hunk_with_removal:
                out.append(f"  @@ base line {old_no}, head line {new_no}")
            continue
        if line.startswith(("---", "+++", "diff --git", "index ", "similarity", "rename ")):
            continue
        if line.startswith("Binary files"):
            out.append("  (binary file changed)")
            removed_total += 1
            continue
        if not in_hunk_with_removal:
            continue
        if line.startswith("-"):
            out.append(f"  -{old_no}: {line[1:]}")
            old_no += 1
            removed_total += 1
        elif line.startswith("+"):
            out.append(f"  +{new_no}: {line[1:]}")
            new_no += 1
        elif line.startswith("\\"):
            out.append(f"  {line}")
    return out, removed_total


def changed_paths(base: str, head: str | None) -> list[tuple[str, str, str | None]]:
    """(status, old path, new path) for every path that differs, renames detected."""
    args = ["diff", "--name-status", "-z", "--find-renames", "--no-ext-diff", base]
    if head:
        args.append(head)
    fields = git(*args).split("\0")
    out = []
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i]
        if status[0] in "RC":
            out.append((status[0], fields[i + 1], fields[i + 2]))
            i += 3
        else:
            out.append((status[0], fields[i + 1], None))
            i += 2
    return out


def _head_text(head: str | None, path: str) -> str:
    if head is None:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""
    try:
        return git("show", f"{head}:{path}")
    except GitError:
        return ""


def _full_diff_lines(diff_text: str) -> list[str]:
    """Every hunk header and changed line of a unified=0 diff, additions included."""
    out: list[str] = []
    old_no = new_no = 0
    for line in diff_text.splitlines():
        m = _HUNK.match(line)
        if m:
            old_no, new_no = int(m.group(1)), int(m.group(3))
            out.append(f"  @@ base line {old_no}, head line {new_no}")
            continue
        if line.startswith(
            (
                "---",
                "+++",
                "diff --git",
                "index ",
                "similarity",
                "rename ",
                "new file",
                "deleted file",
            )
        ):
            continue
        if line.startswith("-"):
            out.append(f"  -{old_no}: {line[1:]}")
            old_no += 1
        elif line.startswith("+"):
            out.append(f"  +{new_no}: {line[1:]}")
            new_no += 1
        elif line.startswith("Binary files"):
            out.append("  (binary file changed)")
    return out


def build_report(base_ref: str, head: str | None) -> tuple[str, list[FileReport]]:
    merge_base = git("merge-base", base_ref, head or "HEAD").strip()
    reports: list[FileReport] = []
    for status, old, new in changed_paths(merge_base, head):
        diff_range = [merge_base] + ([head] if head else [])
        path = new or old
        if is_guardrail(path) or is_guardrail(old):
            diff = git(
                "diff",
                "--unified=0",
                "--no-color",
                "--no-ext-diff",
                "--find-renames",
                *diff_range,
                "--",
                *([old, new] if new else [old]),
            )
            lines = _full_diff_lines(diff) or [f"  ({status} with no text diff)"]
            reports.append(FileReport(path, "guardrail file changed", lines=lines, changed=1))
            continue
        if status == "A":
            if PurePosixPath(path).name == "conftest.py":
                diff = git(
                    "diff", "--unified=0", "--no-color", "--no-ext-diff", *diff_range, "--", path
                )
                flagged = switch_off_lines(added_lines(diff), _head_text(head, path))
                if flagged:
                    reports.append(
                        FileReport(
                            path,
                            "new conftest.py that switches tests off",
                            lines=flagged,
                            changed=len(flagged),
                        )
                    )
            continue
        if not is_guarded(old):
            continue
        if status == "D":
            content = git("show", f"{merge_base}:{old}")
            lines = content.splitlines()
            rep = FileReport(old, "deleted", changed=max(len(lines), 1))
            rep.lines = [f"  -{n}: {text}" for n, text in enumerate(lines, 1)]
            reports.append(rep)
            continue
        if status == "R":
            diff = git(
                "diff",
                "--unified=0",
                "--no-color",
                "--no-ext-diff",
                "--find-renames",
                *diff_range,
                "--",
                old,
                new,
            )
            lines, removed = parse_hunks(diff)
            rep = FileReport(f"{old} -> {new}", "renamed", lines=lines, changed=removed + 1)
            reports.append(rep)
            continue
        diff = git("diff", "--unified=0", "--no-color", "--no-ext-diff", *diff_range, "--", old)
        lines, removed = parse_hunks(diff)
        if removed:
            reports.append(FileReport(old, "modified", lines=lines, changed=removed))
        elif status == "T" or (not lines and "old mode" in diff):
            reports.append(FileReport(old, "mode or type changed", changed=1))
        flagged = switch_off_lines(added_lines(diff), _head_text(head, old))
        if flagged:
            reports.append(
                FileReport(
                    old, "added lines that switch tests off", lines=flagged, changed=len(flagged)
                )
            )
    return merge_base, reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base", nargs="?", default="main", help="base branch (default: main)")
    parser.add_argument("--head", default="HEAD", help="branch or commit to check (default: HEAD)")
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="compare against the working tree, uncommitted edits included",
    )
    args = parser.parse_args(argv)
    head = None if args.worktree else args.head
    try:
        merge_base, reports = build_report(args.base, head)
    except GitError as exc:
        print(f"check_test_edits: {exc}", file=sys.stderr)
        return 2

    target = "the working tree" if head is None else head
    if not reports:
        print(
            f"OK: no existing test line changed or deleted between {args.base} "
            f"(merge-base {merge_base[:10]}) and {target}; the diff only adds tests."
        )
        return 0
    total = sum(r.changed for r in reports)
    print(
        f"{len(reports)} file(s) to read, {total} flagged line(s) or change(s), between "
        f"{args.base} (merge-base {merge_base[:10]}) and {target}: existing tests changed or "
        f"switched off, or guardrail files touched. Read every hunk below:"
    )
    for rep in reports:
        print(f"\n== {rep.path} ({rep.status})")
        for line in rep.lines:
            print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
