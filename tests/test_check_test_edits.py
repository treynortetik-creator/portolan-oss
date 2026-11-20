"""`scripts/check_test_edits.py` (guardrails plan 2026-09-22, rank 6), run in a throwaway git
repo under `tmp_path`: additions pass, and every kind of edit to an existing test is listed
with its line numbers and fails.

`scripts/` is excluded from the container image and the image has no git history, so this
module skips there."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "check_test_edits.py")
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

BASE_TEST = """\
def test_one():
    assert 1 + 1 == 2


def test_two():
    assert "a" in "abc"
"""


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    return env


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, env=_env(), check=True, capture_output=True, text=True
    ).stdout


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "tests").mkdir(parents=True)
    (r / "portolan" / "db" / "migrations").mkdir(parents=True)
    _git(r, "init", "-q", "-b", "main")
    (r / "tests" / "test_math.py").write_text(BASE_TEST, encoding="utf-8")
    (r / "tests" / "conftest.py").write_text("import pytest\n", encoding="utf-8")
    (r / "portolan" / "db" / "migrations" / "0001_init.sql").write_text(
        "CREATE TABLE a (id int);\n"
    )
    (r / "portolan" / "db" / "migrations" / "MANIFEST.sha256").write_text("aaaa  0001_init.sql\n")
    (r / "app.py").write_text("X = 1\n", encoding="utf-8")
    _commit_all(r, "base")
    _git(r, "checkout", "-q", "-b", "feature")
    return r


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=repo,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_only_additions_pass(repo: Path) -> None:
    (repo / "tests" / "test_math.py").write_text(
        BASE_TEST + "\n\ndef test_three():\n    assert 3 > 2\n", encoding="utf-8"
    )
    (repo / "tests" / "test_new.py").write_text("def test_new():\n    assert True\n")
    (repo / "app.py").write_text("X = 2\n")  # non-test edits are not this script's business
    with (repo / "portolan" / "db" / "migrations" / "MANIFEST.sha256").open("a") as fh:
        fh.write("bbbb  0002_next.sql\n")
    (repo / "portolan" / "db" / "migrations" / "0002_next.sql").write_text("SELECT 1;\n")
    _commit_all(repo, "add tests")
    result = _run(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("OK: no existing test line changed or deleted")


def test_a_weakened_assertion_is_listed_with_line_numbers(repo: Path) -> None:
    edited = BASE_TEST.replace('assert "a" in "abc"', "assert True")
    (repo / "tests" / "test_math.py").write_text(edited, encoding="utf-8")
    _commit_all(repo, "weaken")
    result = _run(repo)
    assert result.returncode == 1
    assert "== tests/test_math.py (modified)" in result.stdout
    assert '  -6:     assert "a" in "abc"' in result.stdout
    assert "  +6:     assert True" in result.stdout


def test_a_deleted_test_line_is_listed(repo: Path) -> None:
    (repo / "tests" / "test_math.py").write_text(
        BASE_TEST.replace("    assert 1 + 1 == 2\n", "    pass\n")
    )
    _commit_all(repo, "gut it")
    result = _run(repo)
    assert result.returncode == 1
    assert "  -2:     assert 1 + 1 == 2" in result.stdout


def test_a_deleted_test_file_is_listed(repo: Path) -> None:
    (repo / "tests" / "test_math.py").unlink()
    _commit_all(repo, "delete")
    result = _run(repo)
    assert result.returncode == 1
    assert "== tests/test_math.py (deleted)" in result.stdout
    assert "  -1: def test_one():" in result.stdout


def test_a_renamed_test_file_is_listed(repo: Path) -> None:
    _git(repo, "mv", "tests/test_math.py", "tests/test_arith.py")
    _commit_all(repo, "rename")
    result = _run(repo)
    assert result.returncode == 1
    assert "== tests/test_math.py -> tests/test_arith.py (renamed)" in result.stdout


def test_an_edited_migration_manifest_line_is_listed(repo: Path) -> None:
    (repo / "portolan" / "db" / "migrations" / "MANIFEST.sha256").write_text(
        "cccc  0001_init.sql\n"
    )
    (repo / "portolan" / "db" / "migrations" / "0001_init.sql").write_text(
        "CREATE TABLE a (id bigint);\n"
    )
    _commit_all(repo, "edit an applied migration and regenerate the manifest")
    result = _run(repo)
    assert result.returncode == 1
    assert "== portolan/db/migrations/MANIFEST.sha256 (modified)" in result.stdout
    assert "== portolan/db/migrations/0001_init.sql (modified)" in result.stdout


def test_worktree_mode_sees_uncommitted_edits(repo: Path) -> None:
    (repo / "tests" / "conftest.py").write_text("")
    assert _run(repo).returncode == 0  # committed HEAD is unchanged
    result = _run(repo, "--worktree")
    assert result.returncode == 1
    assert "  -1: import pytest" in result.stdout


def test_commits_on_main_after_the_branch_point_are_not_counted(repo: Path) -> None:
    _git(repo, "checkout", "-q", "main")
    (repo / "tests" / "test_math.py").write_text(BASE_TEST.replace("1 + 1", "2 + 0"))
    _commit_all(repo, "main moves on")
    _git(repo, "checkout", "-q", "feature")
    (repo / "tests" / "test_extra.py").write_text("def test_x():\n    assert 1\n")
    _commit_all(repo, "branch adds a test")
    result = _run(repo, "main")
    assert result.returncode == 0, result.stdout


def test_bad_base_exits_2(repo: Path) -> None:
    result = _run(repo, "no-such-branch")
    assert result.returncode == 2
    assert "check_test_edits:" in result.stderr


def _on_main(repo: Path, files: dict[str, str]) -> None:
    """Commit files on main and bring the feature branch up to it."""
    _git(repo, "checkout", "-q", "main")
    for rel, body in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(body, encoding="utf-8")
    _commit_all(repo, "base files")
    _git(repo, "checkout", "-q", "feature")
    _git(repo, "merge", "-q", "--ff-only", "main")


def test_an_added_skip_in_an_existing_test_is_listed(repo: Path) -> None:
    (repo / "tests" / "test_math.py").write_text(
        BASE_TEST.replace(
            "def test_one():\n", 'def test_one():\n    import pytest; pytest.skip("later")\n'
        )
    )
    _commit_all(repo, "skip it")
    result = _run(repo)
    assert result.returncode == 1, result.stdout
    assert "== tests/test_math.py (added lines that switch tests off)" in result.stdout
    assert 'pytest.skip("later")' in result.stdout


def test_an_added_xfail_decorator_is_listed(repo: Path) -> None:
    (repo / "tests" / "test_math.py").write_text(
        "import pytest\n\n" + BASE_TEST.replace("def test_two", "@pytest.mark.xfail\ndef test_two")
    )
    _commit_all(repo, "xfail it")
    result = _run(repo)
    assert result.returncode == 1
    assert "@pytest.mark.xfail" in result.stdout


def test_an_early_return_in_a_test_function_is_listed(repo: Path) -> None:
    (repo / "tests" / "test_math.py").write_text(
        BASE_TEST.replace("def test_two():\n", "def test_two():\n    return\n")
    )
    _commit_all(repo, "return early")
    result = _run(repo)
    assert result.returncode == 1
    assert "  +6:     return" in result.stdout


def test_a_return_in_a_new_helper_is_not_flagged(repo: Path) -> None:
    (repo / "tests" / "test_math.py").write_text(
        BASE_TEST
        + "\n\ndef _helper():\n    return 3\n\n\ndef test_three():\n    assert _helper() == 3\n"
    )
    _commit_all(repo, "helper")
    result = _run(repo)
    assert result.returncode == 0, result.stdout


def test_skip_words_inside_strings_and_comments_are_not_flagged(repo: Path) -> None:
    (repo / "tests" / "test_math.py").write_text(
        BASE_TEST + '\n\ndef test_three():\n    msg = "skip it"  # not an xfail\n    assert msg\n'
    )
    _commit_all(repo, "strings")
    assert _run(repo).returncode == 0


def test_a_new_conftest_with_collect_ignore_is_listed(repo: Path) -> None:
    (repo / "tests" / "sub").mkdir()
    (repo / "tests" / "sub" / "conftest.py").write_text('collect_ignore = ["test_a.py"]\n')
    _commit_all(repo, "ignore")
    result = _run(repo)
    assert result.returncode == 1
    assert "== tests/sub/conftest.py (new conftest.py that switches tests off)" in result.stdout


def test_a_new_test_file_that_skips_itself_is_not_flagged(repo: Path) -> None:
    (repo / "tests" / "test_new.py").write_text(
        'import pytest\n\npytest.skip("needs a tool", allow_module_level=True)\n'
    )
    _commit_all(repo, "new skipped module")
    assert _run(repo).returncode == 0


@pytest.mark.parametrize(
    ("path", "base", "head"),
    [
        (
            ".github/workflows/ci.yml",  # path-guard: data
            "jobs:\n  secrets: {}\n  test: {}\n",
            "jobs:\n  test: {}\n",
        ),
        (
            ".claude/hooks/no-secrets.py",  # path-guard: data
            'print("hook")\n',
            'import sys; sys.exit(0)\nprint("hook")\n',
        ),
        ("pyproject.toml", 'addopts = ["-p", "tests.guards"]\n', "addopts = []\n"),
        (".gitleaks.toml", "[extend]\nuseDefault = true\n", "[extend]\nuseDefault = false\n"),
        (".dockerignore", ".env\n", "# nothing\n"),
        ("docker/Dockerfile", "FROM x\n", "FROM y\n"),
        ("scripts/check_test_edits.py", "X = 1\n", "X = 2\n"),  # path-guard: data
        ("scripts/verify.py", "X = 1\n", "X = 1\nY = 2\n"),  # path-guard: data
    ],
)
def test_any_change_to_a_guardrail_file_is_listed(
    repo: Path, path: str, base: str, head: str
) -> None:
    _on_main(repo, {path: base})
    (repo / path).write_text(head, encoding="utf-8")
    _commit_all(repo, "touch a guardrail")
    result = _run(repo)
    assert result.returncode == 1, result.stdout
    assert f"== {path} (guardrail file changed)" in result.stdout


def test_a_new_guardrail_file_is_listed(repo: Path) -> None:
    (repo / ".claude").mkdir()  # path-guard: data
    (repo / ".claude" / "settings.json").write_text("{}\n")  # path-guard: data
    _commit_all(repo, "new settings")
    result = _run(repo)
    assert result.returncode == 1
    assert "== .claude/settings.json (guardrail file changed)" in result.stdout
    assert "  +1: {}" in result.stdout
