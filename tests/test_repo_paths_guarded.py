"""Every test module reaches the directories the image leaves out through `tests/_paths.py`.

The docker image leaves some checkout directories out (`.dockerignore`; the list is read from
it, `tests/_paths.IMAGE_EXCLUDED_DIRS`: `scripts/`, `.claude/`, `.github/`, `infra/`, ...), and
CI's `docker` job runs this suite inside it. A module that builds one of those paths by hand and
loads it at import time fails there at COLLECTION, which stops the job and every deploy. This
check reads the source of every module under `tests/` and fails on a hand-built path, so the
mistake shows in the local suite before CI.

What counts as a hand-built path (outside docstrings, and outside a call to
`require_repo_path` / `needs_repo_path`):
- the bare directory name as a path part: `REPO_ROOT / "scripts"`, `REPO_ROOT / "./scripts"`,
  `Path(".claude", ...)`, `os.path.join(root, ".github")`;
- a string that starts inside one of them: `"scripts/verify.py"`, `".claude/settings.json"`,
  or an f-string part that does right after a placeholder: `f"{ROOT}/scripts/verify.py"`;
- `import scripts...` / `from scripts... import`, and a dotted name handed to
  `importlib.import_module`, `runpy.run_module` or `__import__`: `import_module("scripts.x")`.
Nothing after a module-level `require_repo_path(...)` on a path the image lacks (an excluded
directory, or `.git`) counts: collection skips the module there before that code runs.
A command line that merely mentions a script (`"uv run python scripts/ops/key_status.py"`) and
a bare `.dockerignore` entry (`"scripts/"`) are data, not paths, and pass. A line that really
must build such a path as data can end with `# path-guard: data`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests import _paths
from tests._paths import IMAGE_EXCLUDED_DIRS, needs_repo_path, require_repo_path

TESTS_DIR = Path(__file__).resolve().parent
HELPERS = {"require_repo_path", "needs_repo_path"}
PRAGMA = "# path-guard: data"


def _image_ignored_modules() -> set[str]:
    """Test modules the in-image run never collects (`--ignore=tests/x.py` in
    docker/entrypoint.sh), so a hand-built path in them cannot crash collection there."""
    try:
        text = (TESTS_DIR.parent / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    except OSError:
        return set()
    return set(re.findall(r"--ignore=tests/([\w./-]+\.py)", text))


EXEMPT_FILES = {"_paths.py", Path(__file__).name} | _image_ignored_modules()


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


MODULE_LOADERS = {"import_module", "run_module", "__import__"}


def _starts_inside(value: str) -> bool:
    for d in IMAGE_EXCLUDED_DIRS:
        for prefix in (f"{d}/", f"./{d}/"):
            if value.startswith(prefix) and len(value) > len(prefix):
                return True
    return False


def _bare_dir(value: str) -> bool:
    """`scripts`, `./scripts` or `scripts/`: the directory itself as a path part."""
    name = value[2:] if value.startswith("./") else value
    return name.rstrip("/") in IMAGE_EXCLUDED_DIRS


def _after_placeholder(joined: ast.JoinedStr, node: ast.AST) -> bool:
    values = joined.values
    for i, v in enumerate(values):
        if v is node:
            return i > 0 and isinstance(values[i - 1], ast.FormattedValue)
    return False


def _module_guard_line(tree: ast.Module) -> int | None:
    """Last line of the first top-level `require_repo_path(...)` whose first part is absent
    from the image (an image-excluded directory, or `.git`). Collection skips the whole module
    there, in the image, so nothing after that line runs in it."""
    absent = set(IMAGE_EXCLUDED_DIRS) | {".git"}
    for stmt in tree.body:
        value = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
        if not isinstance(value, ast.Call):
            continue
        fn = value.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        first = value.args[0] if value.args else None
        if (
            name == "require_repo_path"
            and isinstance(first, ast.Constant)
            and first.value in absent
        ):
            return stmt.end_lineno
    return None


def unguarded_paths(source: str, filename: str = "<test>") -> list[str]:
    """`file:line: text` for every hand-built path into an image-excluded directory that
    could run in the image: anything before a module-level `require_repo_path` guard on an
    image-absent path, or anywhere in a module without one."""
    tree = ast.parse(source, filename)
    lines = source.splitlines()
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    docstrings = _docstring_nodes(tree)
    guard_line = _module_guard_line(tree)

    def inside_helper(node: ast.AST) -> bool:
        cur = parents.get(id(node))
        while cur is not None:
            if isinstance(cur, ast.Call):
                fn = cur.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                if name in HELPERS:
                    return True
            cur = parents.get(id(cur))
        return False

    found = []

    def report(node: ast.AST) -> None:
        if guard_line is not None and node.lineno > guard_line:
            return  # the module is skipped in the image before this line runs
        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        if PRAGMA not in line:
            found.append(f"{filename}:{node.lineno}: {line.strip()}")

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "scripts"
        ) or (
            isinstance(node, ast.Import)
            and any(a.name.split(".")[0] == "scripts" for a in node.names)
        ):
            report(node)
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            first = node.args[0] if node.args else None
            if (
                name in MODULE_LOADERS
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.split(".")[0] in IMAGE_EXCLUDED_DIRS
            ):
                report(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings or inside_helper(node):
                continue
            parent = parents.get(id(node))
            as_path_part = _bare_dir(node.value) and (
                (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div))
                or isinstance(parent, ast.Call)
            )
            in_fstring = (
                isinstance(parent, ast.JoinedStr)
                and _after_placeholder(parent, node)
                and node.value.startswith("/")
                and _starts_inside(node.value[1:])
            )
            if as_path_part or in_fstring or _starts_inside(node.value):
                report(node)
    return found


def _test_modules() -> list[Path]:
    return sorted(p for p in TESTS_DIR.rglob("*.py") if p.name not in EXEMPT_FILES)


def test_no_test_module_builds_an_image_excluded_path_by_hand() -> None:
    problems = []
    for path in _test_modules():
        rel = path.relative_to(TESTS_DIR.parent).as_posix()
        problems.extend(unguarded_paths(path.read_text(encoding="utf-8"), rel))
    assert not problems, (
        "These build a path by hand into a directory .dockerignore leaves out of the image "
        "(scripts/, .claude/, .github/, infra/, ...), so the module crashes collection there "
        "and CI's docker job goes red. "
        'Use `from tests._paths import require_repo_path` and `require_repo_path("scripts", '
        '"x.py")` (skips the module when absent), or `needs_repo_path(...)` as a marker:\n'
        + "\n".join(problems)
    )


@pytest.mark.parametrize(
    "source",
    [
        'P = Path(__file__).resolve().parents[1] / "scripts" / "verify.py"\n',
        'P = REPO_ROOT / ".claude" / "hooks" / "no-secrets.py"\n',
        'P = REPO_ROOT / ".github" / "workflows" / "ci.yml"\n',
        'P = Path("scripts/verify.py")\n',
        'P = REPO_ROOT.joinpath("scripts", "x.py")\n',
        'P = os.path.join(ROOT, ".claude", "settings.json")\n',
        'spec = importlib.util.spec_from_file_location("v", "scripts/verify.py")\n',
        'subprocess.run([sys.executable, "scripts/check_test_edits.py"])\n',
        "from scripts.data import build_training_set\n",
        "import scripts.verify\n",
        'P = Path(f"{ROOT}/scripts/verify.py")\n',
        'P = Path(f"{ROOT}/.claude/settings.json")\n',
        'm = importlib.import_module("scripts.verify")\n',
        'm = import_module("scripts.data.build_training_set")\n',
        'runpy.run_module("scripts.data.build_training_set")\n',
        'P = REPO_ROOT / "./scripts" / "verify.py"\n',
        'P = REPO_ROOT / "infra" / "app.py"\n',
        'P = REPO_ROOT / "02-reference-implementation" / "x.py"\n',
        # a module-level guard only covers what comes after it
        'X = (REPO_ROOT / "scripts" / "x.py").read_text()\nrequire_repo_path(".git")\n',
        # a guard on a path the image has does not skip the module there
        'require_repo_path("tests", "_paths.py")\nP = REPO_ROOT / "scripts" / "x.py"\n',
        # a guard inside a function skips that test, not the module
        'def f():\n    require_repo_path(".git")\n\nP = REPO_ROOT / "scripts" / "x.py"\n',
    ],
)
def test_the_check_flags_hand_built_paths(source: str) -> None:
    assert unguarded_paths(source), source


@pytest.mark.parametrize(
    "source",
    [
        'P = require_repo_path("scripts", "verify.py")\n',
        'P = _paths.require_repo_path(".claude", "settings.json")\n',
        'mark = needs_repo_path(".github", "workflows", "ci.yml")\n',
        '"""Tests for `scripts/verify.py`."""\n',
        'def f():\n    """Loads scripts/verify.py."""\n',
        'CMD = "uv run python scripts/ops/key_status.py"\n',
        'for required in (".claude/", "infra/", "scripts/"):\n    pass\n',
        'if parts[i] == ".claude":\n    pass\n',
        'P = REPO_ROOT / "scripts" / "x.py"  # path-guard: data\n',
        'P = Path("/x/repo/.claude/worktrees/agent-1")\n',
        'CMD = f"{sys.executable} scripts/verify.py --target {target}"\n',
        'm = importlib.import_module("portolan.config")\n',
        'P = tmp_path / "var" / "railway"\n',
        # after a module-level guard on an image-absent path the module never runs in the image
        'require_repo_path(".git")\nALLOWED = {"02-reference-implementation/README.md": 3}\n',
        'H = require_repo_path(".claude", "hooks", "x.py")\nP = REPO_ROOT / ".claude" / "s.json"\n',
        '_paths.require_repo_path("scripts")\nimport_module("scripts.verify")\n',
    ],
)
def test_the_check_passes_guarded_paths_and_data(source: str) -> None:
    assert unguarded_paths(source) == [], source


def test_the_directory_list_is_read_from_dockerignore() -> None:
    dockerignore = "# c\n.env\n.venv/\nvar/\nscripts/\n.claude/\ninfra/\n!keep/\n.git/\n*.pyc\n"
    gitignore = ".venv/\nvar/\n.claude/worktrees/\n"
    assert _paths.dockerignored_dirs(dockerignore) == (
        ".venv",
        "var",
        "scripts",
        ".claude",
        "infra",
        ".git",
    )
    assert _paths.image_only_missing_dirs(dockerignore, gitignore) == (
        "scripts",
        ".claude",
        "infra",
    )


def test_the_live_list_covers_the_dirs_the_image_leaves_out() -> None:
    if not (_paths.REPO_ROOT / ".gitignore").exists():
        pytest.skip(".gitignore is not in the image; the fallback list is used there")
    for d in ("scripts", ".claude", ".github", "infra", "02-reference-implementation"):
        assert d in IMAGE_EXCLUDED_DIRS, IMAGE_EXCLUDED_DIRS
    assert "var" not in IMAGE_EXCLUDED_DIRS  # gitignored: a CI checkout lacks it too


def test_the_fallback_matches_the_list_read_from_dockerignore() -> None:
    # Inside the image the fallback is the list, so the flag cases above hold there too.
    if not (_paths.REPO_ROOT / ".gitignore").exists():
        pytest.skip(".gitignore is not in the image; this is checked in a checkout")
    assert set(_paths._FALLBACK_EXCLUDED_DIRS) == set(IMAGE_EXCLUDED_DIRS)


def test_require_repo_path_returns_a_present_path() -> None:
    assert require_repo_path("tests", "_paths.py") == Path(_paths.__file__).resolve()


def test_require_repo_path_skips_when_absent() -> None:
    with pytest.raises(pytest.skip.Exception, match="not in this checkout"):
        require_repo_path("scripts", "no-such-script-anywhere.py")


def test_needs_repo_path_is_a_skipif_marker() -> None:
    absent = needs_repo_path("scripts", "no-such-script-anywhere.py")
    present = needs_repo_path("tests", "_paths.py")
    assert absent.mark.name == present.mark.name == "skipif"
    assert absent.mark.args == (True,) and present.mark.args == (False,)


def test_modules_the_image_run_ignores_are_exempt() -> None:
    if not (TESTS_DIR.parent / "docker" / "entrypoint.sh").exists():
        pytest.skip("no docker/entrypoint.sh")
    assert {"test_infra.py", "test_infra_deploy.py"} <= _image_ignored_modules()
