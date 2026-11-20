"""`scripts/affected_tests.py`: which tests a change selects, when it falls back to the full suite,
and whether its import map still covers every test module in this repo.

Most cases build a small tree under `tmp_path` shaped like the repo (portolan/, scripts/, tests/)
and call `select` on it; the last ones map the real repo."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._paths import REPO_ROOT, require_repo_path

_SCRIPT = require_repo_path("scripts", "affected_tests.py")


def _load():
    spec = importlib.util.spec_from_file_location("affected_tests_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["affected_tests_script"] = module
    spec.loader.exec_module(module)
    return module


at = _load()

BASE_FILES = {
    "portolan/__init__.py": "",
    "portolan/core.py": "VALUE = 1\n",
    "portolan/search/__init__.py": "",
    "portolan/search/query.py": "from . import rank\n",
    "portolan/search/rank.py": "from portolan.core import VALUE\n",
    "portolan/web/__init__.py": "",
    "portolan/web/templating.py": (
        "from pathlib import Path\nTEMPLATES = Path(__file__).parent / 'templates'\n"
    ),
    "portolan/web/templates/page.html": "<p>hi</p>\n",
    "portolan/lint/__init__.py": "",
    "portolan/lint/cli.py": "def main():\n    return 0\n",
    "scripts/tool.py": "from portolan.search.query import rank\n",
    "corpus/out/calls/a.txt": "call\n",
    "tests/conftest.py": "import pytest\n",
    "tests/_paths.py": "def require_repo_path(*parts):\n    return parts\n",
    "tests/helpers.py": "from portolan.web import templating\n",
    "tests/test_query.py": "from portolan.search.query import rank\n",
    "tests/test_core.py": "from portolan import core\n",
    "tests/test_web.py": "from helpers import templating\n",
    "tests/test_lint_cli.py": (
        "import subprocess, sys\n"
        "def test_cli():\n"
        "    subprocess.run([sys.executable, '-m', 'portolan.lint.cli'])\n"
    ),
    "tests/test_tool.py": (
        "from tests._paths import require_repo_path\n"
        "_SCRIPT = require_repo_path('scripts', 'tool.py')\n"
    ),
    "tests/test_corpus.py": (
        "from pathlib import Path\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
        "CORPUS_OUT = REPO_ROOT / 'corpus' / 'out'\n"
        "CALLS = CORPUS_OUT / 'calls'\n"
    ),
    "tests/test_tmp_only.py": (
        "def test_x(tmp_path):\n    (tmp_path / 'portolan' / 'core.py').write_text('')\n"
    ),
}


def _tree(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    root = tmp_path / "repo"
    for rel, text in {**BASE_FILES, **(extra or {})}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _select(root: Path, *changed: str):
    return at.select(at.Graph(root, at._walk(root)), list(changed))


def test_a_module_change_selects_the_tests_that_reach_it_and_no_others(tmp_path):
    sel = _select(_tree(tmp_path), "portolan/search/rank.py")
    assert not sel.full
    # test_query reaches rank through query's relative import; test_tool through the script
    # it names by path. test_core imports only portolan.core, which rank itself imports.
    assert set(sel.tests) == {"tests/test_query.py", "tests/test_tool.py"}


def test_a_shared_module_selects_everything_downstream(tmp_path):
    sel = _select(_tree(tmp_path), "portolan/core.py")
    assert {"tests/test_core.py", "tests/test_query.py", "tests/test_tool.py"} <= set(sel.tests)
    assert "tests/test_web.py" not in sel.tests


def test_a_dotted_module_string_counts_as_an_import(tmp_path):
    sel = _select(_tree(tmp_path), "portolan/lint/cli.py")
    assert set(sel.tests) == {"tests/test_lint_cli.py"}


def test_a_test_that_names_a_data_directory_is_selected_when_a_file_in_it_changes(tmp_path):
    sel = _select(_tree(tmp_path), "corpus/out/calls/a.txt")
    assert set(sel.tests) == {"tests/test_corpus.py"}


def test_a_template_selects_the_tests_of_the_module_that_loads_its_directory(tmp_path):
    sel = _select(_tree(tmp_path), "portolan/web/templates/page.html")
    # through tests/helpers.py, a bare import the way pytest resolves it
    assert set(sel.tests) == {"tests/test_web.py"}


def test_a_path_built_on_tmp_path_is_not_a_repo_reference(tmp_path):
    sel = _select(_tree(tmp_path), "portolan/core.py")
    assert "tests/test_tmp_only.py" not in sel.tests


def test_a_changed_test_file_selects_itself(tmp_path):
    root = _tree(tmp_path, {"tests/test_new.py": "def test_new():\n    assert True\n"})
    assert set(_select(root, "tests/test_new.py").tests) == {"tests/test_new.py"}


def test_a_removed_module_selects_the_tests_that_still_import_it(tmp_path):
    root = _tree(tmp_path, {"tests/test_gone.py": "from portolan.gone import x\n"})
    sel = _select(root, "portolan/gone.py")
    assert set(sel.tests) == {"tests/test_gone.py"}


def test_documentation_alone_runs_nothing(tmp_path):
    sel = _select(_tree(tmp_path), "01-spec/plan.md", "README.md", "docs/deploy.md")
    assert not sel.full and not sel.tests and sel.docs_only


_SCANNER = {"tests/test_no_outside_material.py": "def test_scan():\n    assert True\n"}


def test_documentation_alone_still_runs_the_repo_wide_scanners(tmp_path):
    sel = _select(_tree(tmp_path, _SCANNER), "01-spec/plan.md", "README.md")
    assert not sel.full and sel.docs_only
    assert set(sel.tests) == {"tests/test_no_outside_material.py"}


def test_a_code_change_also_runs_the_repo_wide_scanners(tmp_path):
    sel = _select(_tree(tmp_path, _SCANNER), "portolan/lint/cli.py")
    assert "tests/test_no_outside_material.py" in sel.tests and not sel.docs_only


def test_documentation_plus_a_test_that_names_it_is_not_docs_only(tmp_path):
    root = _tree(
        tmp_path,
        {
            **_SCANNER,
            "tests/test_spec.py": "from tests._paths import require_repo_path\n"
            "S = require_repo_path('01-spec', 'demo.md')\n",
        },
    )
    sel = _select(root, "01-spec/demo.md")
    assert set(sel.tests) == {"tests/test_spec.py", "tests/test_no_outside_material.py"}
    assert not sel.docs_only


def test_documentation_a_test_names_is_not_skipped(tmp_path):
    root = _tree(
        tmp_path,
        {
            "tests/test_spec.py": "from tests._paths import require_repo_path\n"
            "S = require_repo_path('01-spec', 'demo.md')\n"
        },
    )
    sel = _select(root, "01-spec/demo.md")
    assert set(sel.tests) == {"tests/test_spec.py"} and not sel.docs_only


@pytest.mark.parametrize(
    "path",
    [
        "tests/conftest.py",
        "tests/guards.py",
        "tests/_paths.py",
        "portolan/db/migrations/0035_new.sql",
        "portolan/config.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/ci.yml",
        ".claude/hooks/no-secrets.py",
    ],
)
def test_each_guardrail_path_runs_the_full_suite_and_says_why(tmp_path, path):
    sel = _select(_tree(tmp_path), "portolan/core.py", path)
    assert sel.full and not sel.tests
    assert any(r.startswith(path) for r in sel.reasons), sel.reasons


def test_a_file_the_map_cannot_place_runs_the_full_suite(tmp_path):
    sel = _select(_tree(tmp_path), "portolan/core.py", "var/blob.bin")
    assert sel.full
    assert sel.reasons == ["var/blob.bin: the map cannot place it (no module, no file names it)"]


def test_a_module_every_test_reaches_runs_the_full_suite(tmp_path):
    root = _tree(tmp_path, {"tests/conftest.py": "from portolan import core\n"})
    sel = _select(root, "portolan/core.py")
    assert sel.full
    assert sel.reasons == ["portolan/core.py: every test file reaches it through tests/conftest.py"]


def _git(repo: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    ).stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_changed_files_counts_commits_edits_and_new_files_since_the_merge_base(tmp_path):
    root = _tree(tmp_path)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "checkout", "-q", "-b", "feature")
    (root / "portolan" / "core.py").write_text("VALUE = 2\n")
    _git(root, "commit", "-qam", "committed")
    _git(root, "checkout", "-q", "main")
    (root / "README.md").write_text("main moved on\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "main only")
    _git(root, "checkout", "-q", "feature")
    (root / "scripts" / "tool.py").write_text("X = 1\n")  # uncommitted edit
    (root / "tests" / "test_brand_new.py").write_text("def test_it():\n    pass\n")  # untracked
    assert at.changed_files(root, "main", None) == [
        "portolan/core.py",
        "scripts/tool.py",
        "tests/test_brand_new.py",
    ]
    assert at.changed_files(root, "main", "feature") == ["portolan/core.py"]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_cli_prints_the_command_or_the_reason(tmp_path):
    root = _tree(tmp_path)
    _git(root, "init", "-q", "-b", "main")

    def run(*files: str) -> str:
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), "--root", str(root), "--files", *files],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout

    picked = run("portolan/lint/cli.py")
    assert picked.splitlines()[0] == "affected_tests: 1 of 7 test files."
    assert picked.splitlines()[-1] == at.PYTEST_PREFIX + " tests/test_lint_cli.py"
    full = run("tests/conftest.py")
    assert full.startswith("affected_tests: FULL SUITE. Why:\n  - tests/conftest.py changed")
    assert full.splitlines()[-1] == at.PYTEST_PREFIX
    docs = run("01-spec/x.md")
    assert docs.startswith("affected_tests: nothing to run: only documentation changed.")
    assert at.PYTEST_PREFIX not in docs
    (root / "tests" / "test_no_outside_material.py").write_text("def test_x():\n    pass\n")
    docs = run("01-spec/x.md")
    assert docs.startswith("affected_tests: only documentation changed; running just the 1 repo")
    assert docs.splitlines()[-1] == at.PYTEST_PREFIX + " tests/test_no_outside_material.py"


def test_an_import_the_map_cannot_follow_is_reported(tmp_path):
    # tools/ is outside the directories the map reads, so a change there would select nothing
    root = _tree(
        tmp_path,
        {
            "tools/__init__.py": "",
            "tools/helper.py": "X = 1\n",
            "tests/test_orphan.py": "from tools.helper import X\nimport not_installed_anywhere\n",
        },
    )
    graph = at.Graph(root, at._walk(root))
    assert graph.unresolved_imports() == ["tests/test_orphan.py: from tools.helper"]


# ---- the real repo ----


@pytest.fixture(scope="module")
def repo_graph():
    require_repo_path(".git")
    return at.Graph(REPO_ROOT, at.tracked_files(REPO_ROOT))


def test_every_import_in_the_test_suite_resolves_in_the_map(repo_graph):
    """Fails when a test module imports a repo module the map cannot place (a new top-level
    directory, a sys.path trick), which would let a change to it select nothing."""
    assert repo_graph.unresolved_imports("tests/") == []


def test_the_map_holds_every_test_module_pytest_collects(repo_graph):
    on_disk = sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "tests").glob("test_*.py"))
    assert sorted(repo_graph.tests) == on_disk


def test_every_test_that_walks_a_tree_is_declared(repo_graph):
    """A test that walks a directory (rglob, os.walk, git ls-files) sees files it never names;
    it must be in SCANNERS (selected by prefix) or SCANS_NAMED_PATHS (only walks paths it
    names, or tmp_path)."""
    walkers = {
        t
        for t in repo_graph.tests
        if at.TREE_WALK.search((REPO_ROOT / t).read_text(encoding="utf-8"))
    }
    declared = set(at.SCANNERS) | at.SCANS_NAMED_PATHS
    assert walkers - declared == set()
    assert declared - walkers == set(), "stale entries"


@pytest.mark.parametrize(
    ("changed", "test"),
    [
        ("scripts/check_test_edits.py", "tests/test_check_test_edits.py"),
        ("scripts/data/build_training_set.py", "tests/test_world_generator_v2_wp7_scripts.py"),
        ("scripts/demo_questions.txt", "tests/test_verify_script.py"),
        ("infra/deploy/canary.py", "tests/test_infra_deploy.py"),
        ("corpus/truth-table.yaml", "tests/test_bench_truth.py"),
        ("tests/fixtures/connectors/gong/calls.json", "tests/test_connectors_gong.py"),
        ("portolan/web/templates/base.html", "tests/test_web.py"),
        ("portolan/llm/openrouter.py", "tests/test_paid_calls_metered.py"),
    ],
)
def test_tests_that_name_or_load_a_repo_file_are_selected_when_it_changes(
    repo_graph, changed, test
):
    sel = at.select(repo_graph, [changed])
    assert not sel.full, sel.reasons
    assert test in sel.tests


def test_the_tree_walk_pattern_matches_what_it_is_meant_to():
    # built from pieces so this file does not match the pattern itself
    assert at.TREE_WALK.search("root." + "rglob('*.py')")
    assert at.TREE_WALK.search("os." + "walk(root)")
    assert at.TREE_WALK.search('run(["git", "ls-' + 'files"])')
    assert not at.TREE_WALK.search("path.glob('*.sql')")
