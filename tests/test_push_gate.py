"""The local push gate (`.claude/hooks/push_gate.py`), its two callers (the Claude Code hook
`no-secrets.py` and the git `pre-push` script), and the `scripts/ci_local.py` pieces that decide
what a run does. GitHub CI is off, so a push with no passing stamp would deploy unchecked.

Every case uses a throwaway git repository. `.dockerignore` excludes `.claude/`, `scripts/` and
`.github/`, so inside the image this module skips."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests._paths import require_repo_path

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE = require_repo_path(".claude", "hooks", "push_gate.py")
HOOK = require_repo_path(".claude", "hooks", "no-secrets.py")
PRE_PUSH = require_repo_path("scripts", "git-hooks", "pre-push")
CI_LOCAL = require_repo_path("scripts", "ci_local.py")
CI_YML = require_repo_path(".github", "workflows", "ci.yml")
SETTINGS = require_repo_path(".claude", "settings.json")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load("push_gate_under_test", GATE)
ci_local = _load("ci_local_under_test", CI_LOCAL)


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def commit(repo: Path, rel: str, text: str) -> str:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", f"change {rel}")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "checkout"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / ".gitignore").write_text("var/\n")
    commit(r, "portolan/x.py", "X = 1\n")
    return r


def write_stamp(repo: Path, **over) -> Path:
    sha = git(repo, "rev-parse", "HEAD")
    stamp = {
        "sha": sha,
        "tree": git(repo, "rev-parse", "HEAD^{tree}"),
        "clean": True,
        "mode": "full",
        "result": "pass",
        "steps": [{"name": "lint", "result": "pass"}, {"name": "tests", "result": "pass"}],
    }
    stamp.update(over)
    path = repo / "var" / "ci" / f"{sha}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp))
    return path


# --------------------------------------------------------------------------- the gate itself


def test_a_passing_stamp_for_a_clean_head_allows_the_push(repo: Path) -> None:
    write_stamp(repo)
    assert gate.problems(str(repo)) == []


def test_no_stamp_refuses(repo: Path) -> None:
    (problem,) = gate.problems(str(repo))
    assert "no CI stamp" in problem


def test_a_stamp_for_an_older_tree_refuses(repo: Path) -> None:
    write_stamp(repo, tree="0" * 40)
    assert any("different tree" in p for p in gate.problems(str(repo)))


def test_a_stamp_for_an_earlier_commit_does_not_cover_a_new_one(repo: Path) -> None:
    stamp = write_stamp(repo)
    commit(repo, "portolan/y.py", "Y = 2\n")
    (problem,) = gate.problems(str(repo))
    assert "no CI stamp" in problem and stamp.exists()


def test_a_dirty_worktree_refuses(repo: Path) -> None:
    write_stamp(repo)
    (repo / "portolan" / "x.py").write_text("X = 2\n")
    assert any("uncommitted" in p for p in gate.problems(str(repo)))


def test_an_untracked_file_counts_as_dirty(repo: Path) -> None:
    write_stamp(repo)
    (repo / "notes.txt").write_text("scratch")
    assert any("uncommitted" in p for p in gate.problems(str(repo)))


def test_a_failed_stamp_refuses(repo: Path) -> None:
    write_stamp(repo, result="fail")
    assert any("does not say pass" in p for p in gate.problems(str(repo)))


def test_a_stamp_with_a_failed_step_refuses_even_if_the_result_says_pass(repo: Path) -> None:
    write_stamp(repo, steps=[{"name": "tests", "result": "fail"}])
    assert any("does not say pass" in p for p in gate.problems(str(repo)))


def test_a_stamp_from_a_dirty_run_refuses(repo: Path) -> None:
    write_stamp(repo, clean=False)
    assert any("uncommitted changes" in p for p in gate.problems(str(repo)))


def test_an_unreadable_stamp_refuses(repo: Path) -> None:
    write_stamp(repo).write_text("{not json")
    assert any("unreadable" in p for p in gate.problems(str(repo)))


def test_a_pushed_sha_other_than_head_needs_its_own_stamp(repo: Path) -> None:
    first = git(repo, "rev-parse", "HEAD")
    write_stamp(repo)
    commit(repo, "portolan/y.py", "Y = 2\n")
    assert gate.problems(str(repo), first) == []
    assert gate.problems(str(repo), "HEAD")


# --------------------------------------------------------------------------- Claude Code hook


def run_hook(command: str, cwd: Path) -> subprocess.CompletedProcess:
    payload = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(cwd)}
    return subprocess.run(
        [sys.executable, str(HOOK)], input=json.dumps(payload), capture_output=True, text=True
    )


def test_the_hook_blocks_git_push_without_a_stamp(repo: Path) -> None:
    out = run_hook("git push origin main", repo)
    assert out.returncode == 2
    assert "ci_local.py" in out.stderr


def test_the_hook_blocks_a_push_run_with_dash_c_or_after_cd(repo: Path, tmp_path: Path) -> None:
    assert run_hook(f"git -C {repo} push", tmp_path).returncode == 2
    assert run_hook(f"cd {repo} && git push", tmp_path).returncode == 2


def test_the_hook_allows_git_push_with_a_passing_stamp(repo: Path) -> None:
    write_stamp(repo)
    assert run_hook("git push origin main", repo).returncode == 0


def test_the_hook_blocks_git_push_after_the_tree_changes(repo: Path) -> None:
    write_stamp(repo)
    (repo / "portolan" / "x.py").write_text("X = 3\n")
    assert run_hook("git push", repo).returncode == 2


def test_the_hook_still_refuses_any_push_from_an_agent_worktree(repo: Path) -> None:
    wt = repo / ".claude" / "worktrees" / "agent-x"
    wt.mkdir(parents=True)
    write_stamp(repo)
    assert run_hook("git push", wt).returncode == 2


def test_the_hook_does_not_mistake_other_git_commands_for_a_push(repo: Path) -> None:
    assert run_hook("git status", repo).returncode == 0
    assert run_hook("git log --oneline -3", repo).returncode == 0


# --------------------------------------------------------------------------- git pre-push hook


def run_pre_push(repo: Path) -> subprocess.CompletedProcess:
    # The script finds push_gate.py under the checkout, so give the throwaway repo a copy.
    dest = repo / ".claude" / "hooks"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "push_gate.py").write_text(GATE.read_text())
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "gate")
    return run_pre_push_again(repo)


def test_pre_push_refuses_without_a_stamp_and_allows_with_one(repo: Path) -> None:
    refused = run_pre_push(repo)
    assert refused.returncode == 1 and "no CI stamp" in refused.stderr
    write_stamp(repo)
    assert run_pre_push_again(repo).returncode == 0


def run_pre_push_again(repo: Path) -> subprocess.CompletedProcess:
    head = git(repo, "rev-parse", "HEAD")
    return subprocess.run(
        ["sh", str(PRE_PUSH)],
        cwd=repo,
        input=f"refs/heads/main {head} refs/heads/main {'0' * 40}\n",
        capture_output=True,
        text=True,
    )


def test_pre_push_ignores_a_branch_deletion(repo: Path) -> None:
    dest = repo / ".claude" / "hooks"
    dest.mkdir(parents=True)
    (dest / "push_gate.py").write_text(GATE.read_text())
    out = subprocess.run(
        ["sh", str(PRE_PUSH)],
        cwd=repo,
        input=f"(delete) {'0' * 40} refs/heads/old {'a' * 40}\n",
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0


# --------------------------------------------------------------------------- ci_local


def test_a_documentation_only_change_since_the_last_pass_selects_docs_only_mode(
    repo: Path,
) -> None:
    write_stamp(repo)
    commit(repo, "01-spec/notes.md", "# notes\n")
    head = git(repo, "rev-parse", "HEAD")
    mode, why = ci_local.choose_mode(repo, head, force_full=False)
    assert mode == "docs-only", why


def test_a_code_change_since_the_last_pass_selects_full_mode(repo: Path) -> None:
    write_stamp(repo)
    commit(repo, "portolan/y.py", "Y = 2\n")
    head = git(repo, "rev-parse", "HEAD")
    assert ci_local.choose_mode(repo, head, force_full=False)[0] == "full"


def test_no_earlier_pass_selects_full_mode(repo: Path) -> None:
    commit(repo, "01-spec/notes.md", "# notes\n")
    head = git(repo, "rev-parse", "HEAD")
    assert ci_local.choose_mode(repo, head, force_full=False)[0] == "full"


def test_a_failed_earlier_stamp_is_not_a_base_for_docs_only(repo: Path) -> None:
    write_stamp(repo, result="fail")
    commit(repo, "01-spec/notes.md", "# notes\n")
    head = git(repo, "rev-parse", "HEAD")
    assert ci_local.choose_mode(repo, head, force_full=False)[0] == "full"


def test_force_full_beats_docs_only(repo: Path) -> None:
    write_stamp(repo)
    commit(repo, "01-spec/notes.md", "# notes\n")
    head = git(repo, "rev-parse", "HEAD")
    assert ci_local.choose_mode(repo, head, force_full=True)[0] == "full"


def test_a_full_run_stops_at_the_start_when_docker_is_down(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(ci_local, "docker_ready", lambda root: False)
    assert ci_local.main(["--root", str(repo)]) == 1
    assert "colima start" in capsys.readouterr().out
    stamp = json.loads((repo / "var" / "ci" / f"{git(repo, 'rev-parse', 'HEAD')}.json").read_text())
    assert stamp["result"] == "fail" and stamp["steps"][0]["name"] == "docker"
    assert gate.problems(str(repo))  # and that stamp does not open the push gate


# --------------------------------------------------------------------------- wiring


def test_the_workflow_runs_only_when_started_by_hand() -> None:
    workflow = yaml.safe_load(CI_YML.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))  # PyYAML reads a bare `on` as True
    assert set(triggers) == {"workflow_dispatch"}
    assert {"secrets", "test", "docker"} <= set(workflow["jobs"])


def test_the_bash_hook_is_still_registered_and_finds_the_gate_next_to_it() -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    commands = [
        h["command"]
        for entry in settings["hooks"]["PreToolUse"]
        if entry["matcher"] == "Bash"
        for h in entry["hooks"]
    ]
    assert any("no-secrets.py" in c for c in commands)
    assert (HOOK.parent / "push_gate.py").is_file()
