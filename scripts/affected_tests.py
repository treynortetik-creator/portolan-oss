"""Pick the test files a change can affect, so an agent does not run the 12-minute suite per edit.

    uv run python scripts/affected_tests.py              # changes vs main, committed + uncommitted
    uv run python scripts/affected_tests.py --base feature/x
    uv run python scripts/affected_tests.py --run        # run the selection (or the full suite)
    uv run python scripts/affected_tests.py --why        # name the chain that picked each test
    uv run python scripts/affected_tests.py --files portolan/search/query.py

It builds a static map of every Python file under `portolan/`, `scripts/`, `tests/`, `corpus/`
and `infra/`: the modules each file imports (anywhere in the file, relative imports included),
the dotted module names it holds as strings (`import_module`, `-m`, `mock.patch` targets), and
the repo paths it names as strings (`require_repo_path("scripts", "x.py")`, `REPO_ROOT / "corpus"
/ "out"`, `Path(__file__).parent / "templates"`). A test is selected when a changed file is
reachable from it through those edges. Every test module also depends on `tests/conftest.py`
and the `tests.guards` plugin, so a change to anything they import selects every test.

It fails closed. The full suite runs, with the reason printed, when a change touches a file in
FULL_SUITE_TRIGGERS or a file the map cannot place (not a Python module, not named by any file,
not inside a package directory, not documentation). A change to documentation alone runs only
the REPO_WIDE tests, which read every tracked file. CI still runs the full suite on every push; this only decides what an agent runs while
it works. Design: 01-spec/borrowed-workflow-ideas-2026-09-23.md (B2),
01-spec/gbrain-borrow-workflow-2026-09-23.md (A1). Tests: tests/test_affected_tests.py.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent

PYTEST_ENV_UNSET = ("OPENROUTER_API_KEY", "DATABASE_URL")
PYTEST_ENV_SET = {"TZ": "UTC", "PGTZ": "UTC", "PORTOLAN_ROLE": "test"}
PYTEST_PREFIX = (
    "env -u OPENROUTER_API_KEY -u DATABASE_URL TZ=UTC PGTZ=UTC PORTOLAN_ROLE=test uv run pytest -q"
)

CODE_ROOTS = ("portolan", "scripts", "tests", "corpus", "infra")
# Directories whose .py files are copies or generated output, never imported by the suite.
SKIP_PY_UNDER = ("corpus/out/",)
# Extra import roots: pytest puts tests/ on sys.path (no __init__.py there), and the infra tests
# put infra/ there to import `portolan_infra`.
EXTRA_IMPORT_ROOTS = ("tests", "infra")

# A path equal to a key, or under a key ending in "/", runs the full suite.
FULL_SUITE_TRIGGERS = {
    "tests/conftest.py": "shared fixtures every test uses",
    "tests/guards.py": "the suite's guardrail plugin",
    "tests/_paths.py": "the repo-path helper many tests load at collection",
    "portolan/db/migrations/": "every database test applies all migrations",
    "portolan/config.py": "settings every entry point reads",
    "pyproject.toml": "pytest options, dependencies and lint config",
    "uv.lock": "dependency versions",
    ".github/": "CI definition",
    ".claude/hooks/": "agent guardrail hooks",
    "docker/": "the image CI runs the suite in",
    ".dockerignore": "what the image leaves out, which decides which tests skip there",
}

DOC_PREFIXES = (
    "01-spec/",
    "04-brand/",
    "docs/",
    "02-reference-implementation/",
    "03-talos-agent-kit/",
)

# Tests that walk a whole tree instead of naming files: selected when any changed path is under
# one of the prefixes ("" = anywhere except documentation). `tests/test_affected_tests.py` fails
# when a test file walks a tree (rglob, os.walk, git ls-files) and is in neither table.
SCANNERS = {
    "tests/test_no_outside_material.py": ("",),
    "tests/test_repo_paths_guarded.py": ("tests/",),
}
# Walks a tree, but only tmp_path trees or roots it names as a string path the map already reads.
SCANS_NAMED_PATHS = {
    "tests/test_extract_prompt.py",
    "tests/test_paid_calls_metered.py",
    "tests/test_store.py",
    "tests/test_synth_batches.py",
    "tests/test_world_generator_v2_wp7.py",
}
# Scanners that read every tracked file whatever its type (a docs-only push once named the
# employer and failed CI), so they run for every change, documentation included.
REPO_WIDE = tuple(t for t, prefixes in SCANNERS.items() if "" in prefixes)
TREE_WALK = re.compile(r"\.rglob\(|os\.walk\(|\"ls-files\"")

# Basenames too common to select on by name alone; a longer path still matches.
GENERIC_BASENAMES = {
    "__init__.py",
    "__main__.py",
    "app.py",
    "cli.py",
    "main.py",
    "config.py",
    "conftest.py",
    "README.md",
    "manifest.json",
    "index.html",
    "base.html",
}
DOTTED = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
PATHLIKE = re.compile(r"^[\w.@+-]+(?:/[\w.@+-]+)*/?$")


def module_name(rel: str) -> str | None:
    """`portolan/search/query.py` -> `portolan.search.query`; a package's `__init__.py` -> the
    package. None for a path that is not an importable module name."""
    p = PurePosixPath(rel)
    if p.suffix != ".py":
        return None
    parts = list(p.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def _with_parents(name: str) -> set[str]:
    parts = name.split(".")
    return {".".join(parts[:i]) for i in range(1, len(parts) + 1)}


@dataclass
class FileFacts:
    imports: set[str] = field(default_factory=set)  # dotted names, parents included
    import_stmts: list[tuple[str, bool]] = field(default_factory=list)  # (name, is_from)
    dotted_strings: set[str] = field(default_factory=set)
    anchored: set[str] = field(default_factory=set)  # repo-relative paths it builds ("" = root)
    loose: set[str] = field(default_factory=set)  # path-like strings with no known anchor


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _norm(parts: list[str]) -> str | None:
    out: list[str] = []
    for part in "/".join(parts).split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not out:
                return None
            out.pop()
        else:
            out.append(part)
    return "/".join(out)


class _PathEval:
    """Evaluates the path expressions tests and scripts use to reach repo files:
    `Path(__file__).resolve().parents[1] / "corpus" / "out"`, a name assigned such an expression,
    `require_repo_path("scripts", "x.py")`. Returns a repo-relative path, "" for the repo root,
    or None when the expression is not anchored to the repo (tmp_path, a parameter)."""

    def __init__(self, rel: str, tree: ast.Module):
        self.rel = rel
        self.assigned: dict[str, ast.expr] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    self.assigned.setdefault(target.id, node.value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                if isinstance(node.target, ast.Name):
                    self.assigned.setdefault(node.target.id, node.value)
        self._busy: set[str] = set()

    def __call__(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return None
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return None
            if node.id in self.assigned and node.id not in self._busy:
                self._busy.add(node.id)
                try:
                    return self(self.assigned[node.id])
                finally:
                    self._busy.discard(node.id)
            return "" if _rootish(node.id) else None
        if isinstance(node, ast.Attribute):
            if node.attr == "parent":
                return _up(self._file_or(node.value), 0)
            return "" if _rootish(node.attr) else None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if (
                node.value.attr == "parents"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)
            ):
                return _up(self._file_or(node.value.value), node.slice.value)
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = self(node.left)
            right = self._piece(node.right)
            if left is None or right is None:
                return None
            return _norm([left, right])
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in ("resolve", "absolute") and isinstance(node.func, ast.Attribute):
                return self(node.func.value)
            if name in ("require_repo_path", "needs_repo_path"):
                pieces = [self._piece(a) for a in node.args]
                return None if any(p is None for p in pieces) else _norm(pieces)
            if name == "joinpath" and isinstance(node.func, ast.Attribute):
                base = self(node.func.value)
                pieces = [self._piece(a) for a in node.args]
                if base is None or any(p is None for p in pieces):
                    return None
                return _norm([base, *pieces])
            if name == "Path" and len(node.args) == 1:
                return self(node.args[0])
        return None

    def _file_or(self, node: ast.expr) -> str | None:
        """`Path(__file__)` (possibly `.resolve()`d) is this file's path; anything else evaluates."""
        while (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("resolve", "absolute")
        ):
            node = node.func.value
        if (
            isinstance(node, ast.Call)
            and _call_name(node.func) == "Path"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "__file__"
        ):
            return self.rel
        return self(node)

    def _piece(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.assigned:
            value = self.assigned[node.id]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
        return None


def _up(path: str | None, n: int) -> str | None:
    """`Path(path).parents[n]` as a repo-relative path; None above the repo root."""
    if path is None or path == "":
        return None
    ups = PurePosixPath(path).parents
    if n >= len(ups):
        return None
    out = str(ups[n])
    return "" if out == "." else out


def _rootish(name: str) -> bool:
    low = name.lower()
    return ("root" in low or "repo" in low) and "tmp" not in low


def _string_tail(node: ast.BinOp) -> str | None:
    """The trailing string pieces of an unanchored `x / "a" / "b"`: "a/b"."""
    pieces: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
        right = cur.right
        if not (isinstance(right, ast.Constant) and isinstance(right.value, str)):
            break
        pieces.insert(0, right.value)
        cur = cur.left
    return _norm(pieces) if pieces else None


def _tmp_based(node: ast.BinOp) -> bool:
    """`tmp_path / "portolan" / "x.py"` builds a scratch file, not a reference to the repo's."""
    cur: ast.expr = node
    while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
        cur = cur.left
    name = (
        cur.id if isinstance(cur, ast.Name) else cur.attr if isinstance(cur, ast.Attribute) else ""
    )
    return "tmp" in name.lower()


def parse_file(rel: str, text: str) -> FileFacts:
    """The imports, dotted-name strings and path strings in one Python file."""
    facts = FileFacts()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return facts
    pkg_parts = list(PurePosixPath(rel).parent.parts)
    evaluate = _PathEval(rel, tree)
    inner_divs = {
        id(n.left)
        for n in ast.walk(tree)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and isinstance(n.left, ast.BinOp)
    }
    here = PurePosixPath(rel).parent
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                facts.import_stmts.append((alias.name, False))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                facts.import_stmts.append((base, True))
            for alias in node.names:
                if alias.name != "*" and base:
                    facts.imports.add(f"{base}.{alias.name}")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if id(node) in inner_divs:
                continue
            if _tmp_based(node):
                skip.update(id(n) for n in ast.walk(node) if isinstance(n, ast.Constant))
                continue
            path = evaluate(node)
            if path:
                facts.anchored.add(path)
            elif path is None:
                tail = _string_tail(node)
                if tail:
                    facts.loose.add(tail)
        elif isinstance(node, ast.Call) and _call_name(node.func) in (
            "require_repo_path",
            "needs_repo_path",
            "joinpath",
        ):
            path = evaluate(node)
            if path:
                facts.anchored.add(path)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in skip:
                continue
            s = node.value.strip()
            if len(s) > 300 or not s:
                continue
            if DOTTED.match(s):
                facts.dotted_strings.add(s)
            s = s.removeprefix("./").rstrip("/")
            if not PATHLIKE.match(s) or not re.search(r"[A-Za-z0-9]", s) or ".." in s.split("/"):
                continue
            if "/" in s or s.startswith(".") or re.search(r"\w\.\w+$", s):
                facts.loose.add(s)
    for name, _ in facts.import_stmts:
        facts.imports.add(name)
    facts.imports = (
        set().union(*(_with_parents(n) for n in facts.imports)) if facts.imports else set()
    )
    for s in list(facts.loose):
        joined = _norm([str(here), s])  # relative to the file's own directory
        if joined:
            facts.anchored.add(joined)
    return facts


class Graph:
    """The static dependency map of the repo's Python files."""

    def __init__(self, root: Path, files: list[str]):
        self.root = root
        self.files = sorted(files)
        self.py = [f for f in self.files if f.endswith(".py") and _is_mapped_py(f)]
        self.name_of = {f: module_name(f) for f in self.py}
        self.file_of = {n: f for f, n in self.name_of.items() if n}
        self.facts = {f: parse_file(f, _read(root / f)) for f in self.py}
        self.tests = [f for f in self.py if _is_test_file(f)]
        self._rev = self._reverse_edges()

    def resolve(self, importer: str, name: str) -> str | None:
        """The repo file an imported dotted name loads, trying the name as written, next to the
        importer (a script run by path, a tests/ helper) and under the extra import roots."""
        here = ".".join(PurePosixPath(importer).parent.parts)
        for cand in (
            name,
            f"{here}.{name}" if here else name,
            *tuple(f"{r}.{name}" for r in EXTRA_IMPORT_ROOTS),
        ):
            if cand in self.file_of:
                return self.file_of[cand]
        return None

    def _targets(self, importer: str, facts: FileFacts) -> set[str]:
        out: set[str] = set()
        for name in facts.imports:
            f = self.resolve(importer, name)
            if f:
                out.add(f)
        for s in facts.dotted_strings:
            parts = s.split(".")
            for i in range(len(parts), 0, -1):
                f = self.resolve(importer, ".".join(parts[:i]))
                if f:
                    out.add(f)
                    break
        for p in facts.anchored:
            if p in self.name_of:
                out.add(p)
        out.discard(importer)
        return out

    def _reverse_edges(self) -> dict[str, set[str]]:
        rev: dict[str, set[str]] = {f: set() for f in self.py}
        implicit = [f for f in ("tests/conftest.py", "tests/guards.py") if f in self.name_of]
        for f in self.py:
            targets = self._targets(f, self.facts[f])
            if f.startswith("tests/") and f not in implicit:
                targets.update(implicit)
            for t in targets:
                rev.setdefault(t, set()).add(f)
        return rev

    def naming(self, path: str) -> set[str]:
        """Files that name `path` (or a directory holding it) as a string path, or import it
        (for a module that no longer exists: by the module name it had)."""
        out: set[str] = set()
        name = module_name(path)
        base = PurePosixPath(path).name
        for f, facts in self.facts.items():
            if name and (name in facts.imports or name in facts.dotted_strings):
                out.add(f)
                continue
            if any(a == path or (a and path.startswith(a + "/")) for a in facts.anchored):
                out.add(f)
                continue
            for s in facts.loose:
                if s == path or ("/" in s and path.endswith("/" + s)):
                    out.add(f)
                    break
                if s == base and base not in GENERIC_BASENAMES and not base.endswith(".py"):
                    out.add(f)
                    break
        out.discard(path)
        return out

    def package_modules(self, path: str) -> list[str]:
        """The modules beside a non-Python file in its nearest code directory (templates, SQL,
        prompts): the code that most likely loads it. Empty under tests/ or outside code."""
        d = PurePosixPath(path).parent
        while str(d) not in (".", ""):
            if d.parts[0] == "tests":
                return []
            mods = [f for f in self.py if PurePosixPath(f).parent == d]
            if mods:
                return mods
            d = d.parent
        return []

    def reachers(self, start: set[str]) -> dict[str, str]:
        """Every file that depends on any file in `start`, mapped to the file it reached it by."""
        seen: dict[str, str] = {}
        queue = deque(start)
        while queue:
            cur = queue.popleft()
            for dep in self._rev.get(cur, ()):
                if dep not in seen and dep not in start:
                    seen[dep] = cur
                    queue.append(dep)
        return seen

    def is_namespace(self, name: str) -> bool:
        """A directory of modules with no `__init__.py` (`tests`, `scripts`)."""
        return any(n.startswith(name + ".") for n in self.file_of)

    def unresolved_imports(self, under: str = "tests/") -> list[str]:
        """Imports in files under `under` that the map cannot follow although the repo has a
        file or directory by that name: a change there would select nothing. Imports of the
        standard library and installed or optional packages (not in the repo) are skipped."""
        repo_names = _repo_names(self.root)
        out = []
        for f in self.py:
            if not f.startswith(under):
                continue
            for name, is_from in self.facts[f].import_stmts:
                if self.resolve(f, name) or self.is_namespace(name):
                    continue
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or _installed(top):
                    continue
                if top in repo_names:
                    kind = "from" if is_from else "import"
                    out.append(f"{f}: {kind} {name}")
        return out


def _repo_names(root: Path) -> set[str]:
    """Every file stem and directory name in the checkout, outside hidden and virtualenv dirs."""
    names: set[str] = set()
    for _dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        names.update(dirnames)
        names.update(Path(f).stem for f in filenames)
    return names


def _installed(top: str) -> bool:
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    origins = [spec.origin] if spec.origin else list(spec.submodule_search_locations or [])
    # the virtualenv may live inside the checkout (.venv); its site-packages are not repo code
    return any(
        o and ("site-packages" in Path(o).parts or REPO_ROOT not in Path(o).resolve().parents)
        for o in origins
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _is_mapped_py(rel: str) -> bool:
    return rel.split("/", 1)[0] in CODE_ROOTS and not rel.startswith(SKIP_PY_UNDER)


def _is_test_file(rel: str) -> bool:
    p = PurePosixPath(rel)
    return p.parts[0] == "tests" and p.name.startswith("test_") and p.suffix == ".py"


def full_trigger(path: str) -> str | None:
    for key, why in FULL_SUITE_TRIGGERS.items():
        if path == key or (key.endswith("/") and path.startswith(key)):
            return why
    return None


def is_doc(path: str) -> bool:
    return path.startswith(DOC_PREFIXES) or path.endswith(".md")


@dataclass
class Selection:
    full: bool = False
    reasons: list[str] = field(default_factory=list)  # why full, or per-file summary lines
    tests: dict[str, str] = field(default_factory=dict)  # test file -> why
    docs_only: bool = False
    total_tests: int = 0


def select(graph: Graph, changed: list[str]) -> Selection:
    sel = Selection(total_tests=len(graph.tests))
    changed = sorted(set(changed))
    if not changed:
        sel.reasons.append("no changed files")
        return sel
    full_reasons = []
    doc_files = []
    for path in changed:
        why = full_trigger(path)
        if why:
            full_reasons.append(f"{path} changed ({why})")
            continue
        starts: set[str] = set()
        how = ""
        if path in graph.name_of:
            starts = {path}
            how = "module"
        named = graph.naming(path)
        if named:
            starts |= named
            how = how or "named by " + ", ".join(sorted(named)[:3])
        if not starts and path.endswith(".py") and _is_mapped_py(path):
            how = "removed module, nothing imports it"
        if not starts and not path.endswith(".py"):
            mods = graph.package_modules(path)
            if mods:
                starts = set(mods)
                how = f"beside the modules in {PurePosixPath(path).parent}"
        scanners = [
            t
            for t, prefixes in SCANNERS.items()
            if t in graph.name_of
            and any((p == "" and not is_doc(path)) or (p and path.startswith(p)) for p in prefixes)
        ]
        if not starts and not scanners and not how:
            if is_doc(path):
                doc_files.append(path)
                continue
            full_reasons.append(f"{path}: the map cannot place it (no module, no file names it)")
            continue
        picked: dict[str, str] = {}
        reached = graph.reachers(starts)
        for s in starts:
            if _is_test_file(s):
                picked[s] = "changed" if s == path else f"names {path}"
        for dep, via in reached.items():
            if _is_test_file(dep) and dep not in picked:
                picked[dep] = f"via {via}"
        for t in scanners:
            picked.setdefault(t, "walks the tree")
        if path.endswith(".py") and _is_test_file(path) and (graph.root / path).exists():
            picked[path] = "changed"
        if picked and len(picked) == len(graph.tests):
            via_conftest = (
                " through tests/conftest.py" if "tests/conftest.py" in {*reached, *starts} else ""
            )
            full_reasons.append(f"{path}: every test file reaches it{via_conftest}")
            continue
        for t, why in picked.items():
            sel.tests.setdefault(t, f"{path} ({why})")
        noun = "test file" if len(picked) == 1 else "test files"
        sel.reasons.append(f"{path}: {len(picked)} {noun}" + (f" [{how}]" if how else ""))
    if full_reasons:
        sel.full = True
        sel.reasons = full_reasons
        sel.tests = {}
        return sel
    wide = [t for t in REPO_WIDE if t in graph.name_of]
    for path in doc_files:
        sel.reasons.append(f"{path}: documentation, no test names it")
    for path in (p for p in changed if is_doc(p)):
        for t in wide:
            sel.tests.setdefault(t, f"{path} (reads every tracked file, documentation included)")
    sel.docs_only = len(doc_files) == len(changed) and set(sel.tests) <= set(wide)
    return sel


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"affected_tests: git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def changed_files(root: Path, base: str, head: str | None) -> list[str]:
    """Paths changed from the merge-base of `base` and `head` (HEAD plus uncommitted and
    untracked files when `head` is None). Renames count as a delete and an add."""
    tip = head or "HEAD"
    mb = _git(root, "merge-base", base, tip).strip()
    out = set(_git(root, "diff", "--name-only", "--no-renames", mb, tip).split())
    if head is None:
        out |= set(_git(root, "diff", "--name-only", "--no-renames", "HEAD").split())
        out |= set(_git(root, "ls-files", "--others", "--exclude-standard").split())
    return sorted(out)


def tracked_files(root: Path) -> list[str]:
    files = _git(root, "ls-files", "-co", "--exclude-standard", *CODE_ROOTS).splitlines()
    return [f for f in files if (root / f).is_file()]


def pytest_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in PYTEST_ENV_UNSET}
    env.update(PYTEST_ENV_SET)
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="main", help="compare against this ref (default: main)")
    ap.add_argument("--head", help="compare this ref instead of the working tree")
    ap.add_argument("--files", nargs="+", help="treat these paths as the change; skip git")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="tree to map (default: repo)")
    ap.add_argument("--run", action="store_true", help="run the selection with pytest")
    ap.add_argument("--why", action="store_true", help="print why each test was selected")
    ap.add_argument("--list", action="store_true", help="print only the selected test paths")
    ap.add_argument("pytest_args", nargs="*", help="after --, extra pytest arguments")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    changed = args.files if args.files else changed_files(root, args.base, args.head)
    graph = Graph(root, tracked_files(root) if (root / ".git").exists() else _walk(root))
    sel = select(graph, changed)
    tests = sorted(sel.tests)

    if args.list:
        print("\n".join(["tests"] if sel.full else tests))
        return 0

    say = sys.stdout.write
    if sel.full:
        say("affected_tests: FULL SUITE. Why:\n")
        for r in sel.reasons:
            say(f"  - {r}\n")
        cmd_paths: list[str] = []
    elif sel.docs_only and tests:
        say(
            f"affected_tests: only documentation changed; running just the {len(tests)} repo-wide "
            "check(s), tests that read every tracked file whatever its type "
            "(employer name, data-shaped files):\n"
        )
        for r in sel.reasons:
            say(f"  - {r}\n")
        cmd_paths = tests
    elif not tests:
        what = "only documentation changed" if sel.docs_only else "no test reaches the change"
        say(f"affected_tests: nothing to run: {what}. CI still runs the full suite.\n")
        for r in sel.reasons:
            say(f"  - {r}\n")
        return 0
    else:
        say(f"affected_tests: {len(tests)} of {sel.total_tests} test files.\n")
        for r in sel.reasons:
            say(f"  - {r}\n")
        if args.why:
            for t in tests:
                say(f"    {t}  <- {sel.tests[t]}\n")
        cmd_paths = tests
    extra = list(args.pytest_args)
    say(" ".join([PYTEST_PREFIX, *cmd_paths, *extra]) + "\n")
    if not args.run:
        return 0
    sys.stdout.flush()
    cmd = ["uv", "run", "pytest", "-q", *cmd_paths, *extra]
    return subprocess.run(cmd, cwd=root, env=pytest_env()).returncode


def _walk(root: Path) -> list[str]:
    out = []
    for top in CODE_ROOTS:
        base = root / top
        if base.is_dir():
            out += [
                str(p.relative_to(root).as_posix())
                for p in base.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            ]
    return out


if __name__ == "__main__":
    sys.exit(main())
