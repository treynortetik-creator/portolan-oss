"""`.claude/hooks/no-secrets.py`, the PreToolUse hook on Bash, Monitor, Read and Grep
(guardrails plan 2026-09-22, rank 2; review 2026-09-22 ranks 1-4 and 19). Every case feeds a
real JSON payload to the script in a subprocess, the way Claude Code runs it, and checks the
exit code: 2 blocks (stderr goes back to the agent), 0 allows. Commands are never run.

Key-shaped literals are assembled at run time (`"AKIA" + ...`) so this file never contains
one: gitleaks would flag it, and the hook itself would refuse a `cat` of it.

`.dockerignore` excludes `.claude/`, so inside the image this module skips (same reason as
`tests/test_ci_workflow.py`)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._paths import require_repo_path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK = require_repo_path(".claude", "hooks", "no-secrets.py")
SETTINGS = require_repo_path(".claude", "settings.json")

AWS_KEY = "AKIA" + "Z7Q2" * 4
OPENROUTER_KEY = "sk-or-" + "v1-" + "0a1b2c3d" * 8
ANTHROPIC_KEY = "sk-ant-" + "api03-" + "Xy9_" * 10


def run_hook(payload, python: str = sys.executable) -> subprocess.CompletedProcess:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [python, str(HOOK)], input=data, capture_output=True, text=True, timeout=30
    )


def bash(command: str, cwd: Path | str = REPO_ROOT, tool: str = "Bash") -> dict:
    return {"tool_name": tool, "tool_input": {"command": command}, "cwd": str(cwd)}


@pytest.fixture(scope="module")
def main_checkout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stand-in for the main checkout: a .env at the top (fake content), var/railway, code, and
    an agent worktree under .claude/worktrees/ with no .env, the way `git worktree add` leaves
    one."""
    root = tmp_path_factory.mktemp("checkout") / "portolan"
    (root / "portolan").mkdir(parents=True)
    (root / "portolan" / "app.py").write_text("X = 1\n")
    (root / ".env").write_text("EXAMPLE_SETTING=1\n")
    (root / ".env.example").write_text("EXAMPLE_SETTING=\n")
    (root / "var" / "railway").mkdir(parents=True)
    wt = root / ".claude" / "worktrees" / "agent-test"  # path-guard: data
    (wt / "portolan").mkdir(parents=True)
    (wt / "portolan" / "app.py").write_text("X = 1\n")
    return root


@pytest.fixture(scope="module")
def worktree(main_checkout: Path) -> Path:
    return main_checkout / ".claude" / "worktrees" / "agent-test"  # path-guard: data


DENY = [
    # reading .env
    "cat .env",
    "cat ./.env",
    'cat "$PWD/.env"',
    "cat /Users/someone/project/.env",
    "head -n 5 .env",
    "tail .env",
    "less .env",
    "more .env",
    "bat .env",
    "nl .env",
    "sed -n 1p .env",
    "awk -F= '{print $2}' .env",
    "grep OPENROUTER .env",
    "grep -i key .env | cut -d= -f2",
    "cut -d= -f2 .env",
    "strings .env",
    "xxd .env",
    "od -c .env",
    "sort .env",
    "cat .env.local",
    "cat < .env",
    "< .env cat",
    'cat .e""nv',
    "cat .e\\nv",
    "cat $'\\x2eenv'",
    "cat .env*",
    "cat {.env,README.md}",
    "cp .env /tmp/x",
    "diff .env .env.example",
    "find . -name .env -exec cat {} \\;",
    "echo .env | xargs cat",
    "F=.env; cat $F",
    "cat $(echo .env)",
    'echo "$(cat .env)"',
    'while read -r line; do echo "$line"; done < .env',
    "ls\ncat .env",
    "true && cat .env",
    "bash -c 'cat .env'",
    'sh -c "head .env"',
    "eval 'cat .env'",
    "sudo cat .env",
    "\\cat .env",
    "/bin/cat .env",
    "command cat .env",
    "env -u X cat .env",
    "uv run python -c \"print(open('.env').read())\"",
    "python3 -c \"from pathlib import Path; print(Path('.env').read_text())\"",
    "python3 - <<'EOF'\nprint(open('.env').read())\nEOF",
    "grep -rn OPENROUTER_API_KEY .",
    "grep -r DATABASE_URL",
    # recursive grep over a tree holding .env prints its matching lines, whatever the pattern
    "grep -rn 'def connect' .",
    "grep -rn OPENROUTER .",
    "grep -rni railway .",
    "rg -uu OPENROUTER",
    "grep -rn MAX_TOKENS .",
    # printing the environment
    "printenv",
    "printenv OPENROUTER_API_KEY",
    "env",
    "env | grep KEY",
    "env -u OPENROUTER_API_KEY",
    "export -p",
    "export",
    "declare -x",
    "declare -p OPENROUTER_API_KEY",
    "typeset",
    "set",
    "set | grep KEY",
    "(set)",
    "ps eww",
    "cat /proc/self/environ",
    "python3 -c 'import os; print(os.environ)'",
    "python3 -c 'import os; print(dict(os.environ))'",
    "node -e 'console.log(process.env)'",
    # railway
    "railway variables",
    "railway variables --kv",
    "railway variables --kv | grep DATABASE",
    "railway variables --json | cut -d= -f1",
    "railway variables --kv | tee vars.txt | cut -d= -f1",
    "railway variables --kv > vars.txt",
    "railway run printenv",
    "railway run -- env",
    "railway run --service Postgres -- sh -c 'echo $PGPASSWORD'",
    # echo / printf of a secret variable
    "echo $OPENROUTER_API_KEY",
    'echo "${DATABASE_URL}"',
    "printf '%s' \"$PGPASSWORD\"",
    "echo ${TYPESAFE_API_KEY:0:6}",
    'echo "key is $ANTHROPIC_API_KEY" >&2',
    "print -r -- $PORTOLAN_SESSION_SECRET",
    'cat <<< "$OPENROUTER_API_KEY"',
    "cat <<EOF\n$GITHUB_TOKEN\nEOF",
    "python3 -c 'import os; print(os.environ[\"OPENROUTER_API_KEY\"])'",
    "uv run python -c 'from portolan.config import get_settings; print(get_settings().openrouter_api_key)'",
    "python3 -c \"print('$OPENROUTER_API_KEY')\"",
    'curl -v -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/key',
    "gh auth token",
    # xtrace while sourcing prints every assignment
    "set -x; . ./.env; set +x",
    "bash -xc '. ./.env; true'",
    # the live keys in var/railway/*key*.txt get the same deny verbs as .env
    "cat var/railway/org-key.txt",
    "cat var/railway/mcp-key.txt",
    "cat ./var/railway/org-key.txt",
    "cat /Users/someone/project/var/railway/mcp-key.txt",
    "head -c 8 var/railway/org-key.txt",
    "tail var/railway/mcp-key.txt",
    "less var/railway/org-key.txt",
    "sed -n 1p var/railway/org-key.txt",
    "awk 1 var/railway/org-key.txt",
    "grep . var/railway/org-key.txt",
    "cut -c1-4 var/railway/org-key.txt",
    "strings var/railway/mcp-key.txt",
    "xxd var/railway/mcp-key.txt",
    "od -c var/railway/mcp-key.txt",
    "sort var/railway/mcp-key.txt",
    "cp var/railway/org-key.txt /tmp/k",
    "cat < var/railway/org-key.txt",
    "cat var/railway/*",
    "cat var/rail*/org*",
    "cat var/railway/{org,mcp}-key.txt",
    "cd var/railway && cat org-key.txt",
    'K=$(cat var/railway/mcp-key.txt); curl -H "Authorization: Bearer $K" http://localhost:8000/mcp',
    ". var/railway/org-key.txt",
    "grep -r . var/railway",
    "grep -rn '' var/",
    "find var/railway -type f -exec cat {} +",
    "python3 -c \"print(open('var/railway/org-key.txt').read())\"",
    "python3 -c \"import os; open('/dev/stdout', 'w').write(os.environ['OPENROUTER_API_KEY'])\"",
    "python3 -c \"import sys, os; sys.stdout.write(os.environ['PGPASSWORD'])\"",
    "bash -c 'cat var/railway/mcp-key.txt'",
    # literal keys
    f"export AWS_ACCESS_KEY_ID={AWS_KEY}",
    f"curl -H 'Authorization: Bearer {OPENROUTER_KEY}' https://openrouter.ai/api/v1/key",
    f"ANTHROPIC_API_KEY={ANTHROPIC_KEY} uv run python x.py",
]

ALLOW = [
    # credential files outside the repo are readable (the maintainer, 2026-09-29)
    "cat ~/.docker/config.json",
    "cat ~/.aws/credentials",
    "cat ~/.pgpass",
    # the exact commands in daily use (CLAUDE.md "Run it")
    "env -u OPENROUTER_API_KEY -u DATABASE_URL TZ=UTC PGTZ=UTC PORTOLAN_ROLE=test uv run pytest -q",
    "env -u OPENROUTER_API_KEY -u DATABASE_URL TZ=UTC PGTZ=UTC PORTOLAN_ROLE=test uv run pytest -q tests/test_web.py",
    "set -a; . ./.env; set +a; uv run python scripts/ops/key_status.py",
    "set -a && source .env && set +a && uv run python -m portolan.db status",
    "railway run --service Postgres -- sh -c '... \"postgresql://$PGUSER:$PGPASSWORD@$RAILWAY_TCP_PROXY_DOMAIN:$RAILWAY_TCP_PROXY_PORT/$PGDATABASE\" ...'",
    'railway run --service Postgres -- sh -c \'psql "postgresql://$PGUSER:$PGPASSWORD@$RAILWAY_TCP_PROXY_DOMAIN:$RAILWAY_TCP_PROXY_PORT/$PGDATABASE" -c "select count(*) from notes"\' | grep -viE "postgres(ql)?://|password"',
    "railway variables --kv | cut -d= -f1",
    "railway variables --service Postgres --kv | cut -d '=' -f 1 | sort",
    "grep -c . .env",
    "grep -c OPENROUTER_API_KEY .env",
    "grep -q '^OPENROUTER_API_KEY=' .env && echo present",
    "cat .env.example",
    "cp .env.example .env",
    "ls -la .env",
    "test -f .env && echo exists",
    '[ -n "${OPENROUTER_API_KEY:+x}" ] && echo set',
    "echo ${#OPENROUTER_API_KEY}",
    "uv run python scripts/ops/key_status.py",
    'echo "$PATH"',
    "printf '%s\\n' \"$HOME\"",
    "export PORTOLAN_ROLE=test",
    "declare -x FOO=bar",
    "set -euo pipefail; uv run pytest -q",
    "uv run ruff check . && uv run ruff format --check .",
    "grep -rn OPENROUTER_API_KEY portolan/ scripts/",
    "rg OPENROUTER_API_KEY",
    "grep -rn --exclude='.env*' OPENROUTER_API_KEY .",
    "echo 'remember to fill in .env'",
    'echo "KEY=value" >> .env',
    "git status --short",
    "git check-ignore .env",
    "git commit -m \"$(cat <<'EOF'\nHook: block cat .env and echo $OPENROUTER_API_KEY\n\nCo-Authored-By: x\nEOF\n)\"",
    "python3 -c 'import os; print(sorted(os.environ))'",
    "python3 -c \"import os; print('OPENROUTER_API_KEY' in os.environ)\"",
    "python3 -c \"import json; print(json.load(open('var/x.json')).keys())\"",
    "ps aux | grep uvicorn",
    "ps -ef | head",
    "railway logs --service web | tail -50",
    'curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models | head -c 200',
    "gh pr view 12",
    # var/railway key files: counts and existence checks, and scripts that use the keys
    "sh var/ops/mcp_probe.sh",
    "bash var/ops/mcp_probe.sh",
    "var/ops/deploy_watch_only.sh 4023fad",
    "wc -c var/railway/org-key.txt",
    "test -s var/railway/mcp-key.txt && echo present",
    "grep -c . var/railway/org-key.txt",
    "grep -q . var/railway/mcp-key.txt && echo ok",
    "ls -la var/railway/",
    "cat var/railway/project.txt",
    "cp /tmp/newkey.txt var/railway/org-key.txt",
    # an edit script that writes files whose names mention secrets is not printing a secret
    "python3 - <<'PY'\nimport json\np = '.claude/settings.json'\nd = json.load(open(p))\n"
    "open(p, 'w').write(json.dumps(d))\nprint('tests/test_no_secrets_hook.py', 'ok')\nPY",
    "python3 -c \"s = open('tests/test_no_secrets_hook.py').read(); "
    "open('/tmp/x.py', 'w').write(s.replace('token', 'tok'))\"",
]


def _case_id(command: str) -> str:
    # Test ids land in .pytest_cache and in reports; keep the key-shaped literals out of them.
    for literal, name in (
        (AWS_KEY, "<aws-key>"),
        (OPENROUTER_KEY, "<openrouter-key>"),
        (ANTHROPIC_KEY, "<anthropic-key>"),
    ):
        command = command.replace(literal, name)
    return command


@pytest.mark.parametrize("command", DENY, ids=_case_id)
def test_hook_blocks(command: str, main_checkout: Path) -> None:
    result = run_hook(bash(command, main_checkout))
    assert result.returncode == 2, f"not blocked: {command!r}\nstderr: {result.stderr}"
    assert "no-secrets hook blocked this command" in result.stderr
    assert "Instead:" in result.stderr


@pytest.mark.parametrize("command", ALLOW)
def test_hook_allows(command: str, main_checkout: Path) -> None:
    result = run_hook(bash(command, main_checkout))
    assert result.returncode == 0, f"blocked: {command!r}\nstderr: {result.stderr}"
    assert result.stderr == ""


def test_blocked_message_never_repeats_a_literal_key() -> None:
    result = run_hook(bash(f"echo {OPENROUTER_KEY}"))
    assert result.returncode == 2
    assert OPENROUTER_KEY not in result.stderr
    assert "OpenRouter key" in result.stderr


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "not json",
        "[1, 2, 3]",
        "null",
        '{"tool_name": "Bash"}',
        '{"tool_name": "Bash", "tool_input": "cat .env"}',
        '{"tool_name": "Bash", "tool_input": {"command": 42}}',
        '{"tool_name": "Bash", "tool_input": {"command": ""}}',
        '{"tool_name": "Bash", "tool_input": {"command": "cat .env"',  # truncated JSON
    ],
)
def test_malformed_payload_fails_open(payload: str) -> None:
    result = run_hook(payload)
    assert result.returncode == 0
    assert result.stderr == ""


def test_tools_it_does_not_check_pass() -> None:
    # Bash, Monitor, Read and Grep are checked; anything else, or a Read with no file_path,
    # passes untouched.
    assert (
        run_hook({"tool_name": "WebFetch", "tool_input": {"url": "https://x/.env"}}).returncode == 0
    )
    payload = {"tool_name": "Read", "tool_input": {"command": "cat .env"}}
    assert run_hook(payload).returncode == 0


def test_unterminated_quote_is_still_checked() -> None:
    # Lenient lexing: an unclosed quote swallows the rest as one word instead of crashing
    # into fail-open, so the `cat .env` before it is still seen.
    assert run_hook(bash("cat .env 'unclosed")).returncode == 2


@pytest.mark.skipif(not Path("/usr/bin/python3").exists(), reason="no system python3")
def test_hook_runs_on_the_system_python() -> None:
    # settings.json runs it as `python3`, which on a fresh Mac is Xcode's 3.9.
    assert run_hook(bash("cat .env"), python="/usr/bin/python3").returncode == 2
    assert run_hook(bash("cat .env.example"), python="/usr/bin/python3").returncode == 0


def test_settings_registers_the_hook_on_bash() -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    entries = settings["hooks"]["PreToolUse"]
    bash_entries = [e for e in entries if e.get("matcher") == "Bash"]
    assert bash_entries, "no PreToolUse entry with matcher 'Bash'"
    commands = [
        h["command"] for e in bash_entries for h in e["hooks"] if h.get("type") == "command"
    ]
    assert any("no-secrets.py" in c for c in commands)


def test_registered_command_runs_the_hook() -> None:
    # Run the exact command string from settings.json through a shell, with the variable
    # Claude Code sets, so a typo in the path or interpreter fails here and not silently.
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    command = next(
        h["command"]
        for e in settings["hooks"]["PreToolUse"]
        if e.get("matcher") == "Bash"
        for h in e["hooks"]
        if "no-secrets.py" in h.get("command", "")
    )
    if shutil.which("python3") is None:
        pytest.skip("no python3 on PATH")
    result = subprocess.run(
        ["sh", "-c", command],
        input=json.dumps(bash("printenv")),
        capture_output=True,
        text=True,
        timeout=30,
        env={"CLAUDE_PROJECT_DIR": str(REPO_ROOT), "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
    )
    assert result.returncode == 2, result.stderr


# --------------------------------------------------------------------------- review 2026-09-22

GITHUB_TOKEN = "gh" + "p_" + "A1b2C3d4" * 5
OPENAI_KEY = "sk-" + "proj-" + "Ab12Cd34" * 4

# The review's hook probe (01-spec/reviews/review-guardrails-2026-09-22.md section 2), the
# should-block cases, run against a main checkout that holds a .env. Two are left out on purpose:
# `ls -a | grep env | xargs cat` and `cat "$(ls -a | grep '^.env$')"` build the file name at run
# time, which command text cannot show (the hook's docstring says so).
REVIEW_DENY = [
    "grep -rn PORTOLAN_ORG .",
    "find . -name '*.env*' | xargs grep KEY",
    "uv run dotenv list",
    "uv run python -m dotenv list",
    "uv run python -m dotenv get OPENROUTER_API_KEY",
    "node -p process.env",
    "node -p process.env.OPENROUTER_API_KEY",
    "ruby -e 'p ENV'",
    "awk 'BEGIN{for(k in ENVIRON) print k\"=\"ENVIRON[k]}'",
    "jq -n env",
    "jq -rn '$ENV.OPENROUTER_API_KEY'",
    "python3 -c 'import os; os.system(\"env\")'",
    "python3 -c \"import subprocess; subprocess.run(['printenv'])\"",
    "uv run python -c 'from portolan.config import get_settings; print(get_settings().__dict__)'",
    "uv run python -c 'from portolan.config import get_settings; s = get_settings(); print(s)'",
    "K=$OPENROUTER_API_KEY; echo $K",
    'K="$OPENROUTER_API_KEY"; curl -sv -H "Authorization: Bearer $K" https://openrouter.ai/api/v1/key',
    "printf -v x '%s' \"$OPENROUTER_API_KEY\"; echo $x",
    "railway run --service Postgres -- sh -c 'echo $DATABASE_PUBLIC_URL'",
    "echo $PORTOLAN_DSN",
    "printf '%s' \"$PGPASS\"",
    "railway ssh --service web -- env",
    "railway ssh --service web printenv DATABASE_URL",
    "railway ssh --service web 'cat /app/.env'",
    "railway run --service web -- node -p process.env.DATABASE_URL",
    "typeset -m '*KEY*'",
    "n=OPENROUTER_API_KEY; print -r -- ${(P)n}",
    "printf 'protocol=https\\nhost=github.com\\n' | git credential fill",
    "aws configure export-credentials",
    "aws configure get aws_secret_access_key",
    "cat ~/.railway/config.json",
    "cat .envrc",
    f"echo {GITHUB_TOKEN}",
    f"echo {OPENAI_KEY}",
    "docker inspect portolan-web",
    "set -a; . ./.env; set +a; uv run pytest -q",
    "set -a; . ./.env; set +a; uv run python -m pytest -q",
    "uv run --env-file .env pytest -q",
]

# Daily use that must keep working in the main checkout (the review's allow list, plus the
# forms its false blocks pointed at).
REVIEW_ALLOW = [
    "uv run python -c 'from portolan.config import get_settings; print(get_settings().max_tokens)'",
    "uv run python -c 'from portolan.config import get_settings; print(get_settings().chat_model)'",
    "python3 - <<'EOF'\nimport re\nprint(re.compile(r'\\.env').pattern)\nEOF",
    "grep -n '^\\.env' .gitignore",
    "grep -rn --include='*.py' OPENROUTER .",
    "rg -uu -g '!.env*' OPENROUTER",
    "rg OPENROUTER_API_KEY",
    "grep -rn max_tokens portolan/",
    "railway run --service Postgres -- sh -c 'echo $PGUSER@$PGHOST'",
    "docker inspect -f '{{.State.Status}}' portolan-web",
    "echo $PGUSER",
    "git log -S OPENROUTER_API_KEY --oneline",
    "gitleaks detect --redact --no-banner",
    "sh var/ops/mcp_probe.sh",
    "var/ops/deploy_watch_only.sh 369309c",
    "railway logs --service web | tail -50",
    "git push origin feature/x",
    "set -a; . ./.env; set +a; uv run python scripts/ops/key_status.py",
]


def _review_id(command: str) -> str:
    return command.replace(GITHUB_TOKEN, "<github-token>").replace(OPENAI_KEY, "<openai-key>")


@pytest.mark.parametrize("command", REVIEW_DENY, ids=_review_id)
def test_review_probe_commands_are_blocked(command: str, main_checkout: Path) -> None:
    result = run_hook(bash(command, main_checkout))
    assert result.returncode == 2, f"not blocked: {command!r}\nstderr: {result.stderr}"
    assert "could not be checked" not in result.stderr  # blocked by a rule, not the fallback


@pytest.mark.parametrize("command", REVIEW_ALLOW)
def test_review_daily_use_is_allowed(command: str, main_checkout: Path) -> None:
    result = run_hook(bash(command, main_checkout))
    assert result.returncode == 0, f"blocked: {command!r}\nstderr: {result.stderr}"


@pytest.mark.parametrize(
    "command",
    ["grep -rn 'def connect' .", "grep -rn OPENROUTER_API_KEY .", "grep -rni railway ."],
)
def test_recursive_grep_is_fine_where_no_env_file_is(command: str, worktree: Path) -> None:
    assert run_hook(bash(command, worktree)).returncode == 0


def test_recursive_grep_finds_a_nested_env(tmp_path: Path) -> None:
    (tmp_path / "svc" / "deploy").mkdir(parents=True)
    (tmp_path / "svc" / "deploy" / ".env.production").write_text("X=1\n")
    assert run_hook(bash("grep -rn X .", tmp_path)).returncode == 2
    assert run_hook(bash("grep -rn X svc/", tmp_path)).returncode == 2
    assert run_hook(bash("grep -rn --exclude='.env*' X .", tmp_path)).returncode == 0


# Agents never touch the live database and never deploy: refused from a worktree only.
WORKTREE_DENY = [
    "railway logs --service web | tail -50",
    "railway status",
    'railway run --service Postgres -- sh -c \'psql "postgresql://$PGUSER:$PGPASSWORD@$RAILWAY_TCP_PROXY_DOMAIN:$RAILWAY_TCP_PROXY_PORT/$PGDATABASE" -c "select 1"\'',
    "railway variables --kv | cut -d= -f1",
    "railway up",
    "/opt/homebrew/bin/railway deployment list",
    "uv run railway status",
    "npx @railway/cli up",
    "sh -c 'railway up'",
    "timeout 30 railway logs",
    "git push",
    "git push origin main",
    "git -C /Users/someone/portolan push origin main",
    "git -c core.hooksPath=/dev/null push --force",
    "cd /Users/someone/portolan && git push",
    "gh pr merge 12 --squash",
    "var/ops/deploy_watch_only.sh 369309c",
    "sh var/ops/confirm_live.sh confirm 12",
    "nohup sh var/ops/label_run.sh > var/logs/x.log 2>&1 &",
    "bash /Users/someone/portolan/var/ops/regen_w36.sh",
    "python3 /Users/someone/portolan/var/ops/backfill.py",
    "cp /Users/someone/portolan/var/ops/research_live.sh /tmp/r.sh",
    "psql 'postgresql://postgres@shinkansen.proxy.rlwy.net:41234/railway' -c 'select 1'",
    "python3 -c \"import subprocess; subprocess.run(['git', 'push'])\"",
]
WORKTREE_ALLOW = [
    "git status --short",
    "git log --oneline -5",
    "git diff main...HEAD -- tests/",
    "git add -A && git commit -m 'work'",
    "gh pr view 12",
    "gh run list --limit 1",
    "cat var/ops/README.md",
    "ls var/ops",
    "grep -n railway docs/deploy-railway.md",
    "uv run pytest -q tests/test_guards.py",
    "echo 'the orchestrator runs railway'",
]


@pytest.mark.parametrize("command", WORKTREE_DENY)
def test_worktree_agents_cannot_touch_live_or_deploy(command: str, worktree: Path) -> None:
    result = run_hook(bash(command, worktree))
    assert result.returncode == 2, f"not blocked: {command!r}\nstderr: {result.stderr}"
    assert "from an agent worktree" in result.stderr
    assert "Agents never touch the live database and never deploy" in result.stderr


@pytest.mark.parametrize(
    "command", [*WORKTREE_DENY[:4], "git push origin main", "gh pr merge 12 --squash"]
)
def test_main_checkout_keeps_live_and_deploy(command: str, main_checkout: Path) -> None:
    assert run_hook(bash(command, main_checkout)).returncode == 0, command


@pytest.mark.parametrize("command", WORKTREE_ALLOW)
def test_worktree_agents_keep_everyday_commands(command: str, worktree: Path) -> None:
    result = run_hook(bash(command, worktree))
    assert result.returncode == 0, f"blocked: {command!r}\nstderr: {result.stderr}"


# Exam test answers stay out of agents' reach: `evals/locked/` is refused from a worktree only
# (01-spec/chat-exam-2026-09-29.md).
LOCKED_FILE = "evals/locked/chat-exam-test.json.gz"
LOCKED_DENY = [
    f"cat {LOCKED_FILE}",
    f"zcat {LOCKED_FILE}",
    f"gunzip -c {LOCKED_FILE} | head -50",
    f"zgrep -i renewal {LOCKED_FILE}",
    f"python3 -c \"import gzip; print(gzip.open('{LOCKED_FILE}').read())\"",
    f"uv run python -c \"import gzip; print(gzip.open('./{LOCKED_FILE}').read())\"",
    f"python3 - < {LOCKED_FILE}",
    f"cp {LOCKED_FILE} /tmp/x.gz",
    f"git show HEAD:{LOCKED_FILE}",
    "git log -p -- evals/locked",
    "git diff main -- evals/locked/",
    "cd evals/locked && ls",
    "cat var/evals/locked/chat-exam/20260929T120000Z-agent/test-transcripts.json.gz",
    "cat /Users/someone/portolan/var/evals/locked/chat-exam/run/test-transcripts.json.gz",
]
LOCKED_ALLOW = [
    "ls evals/locked",
    f"wc -c {LOCKED_FILE}",
    f"stat {LOCKED_FILE}",
    f"test -f {LOCKED_FILE} && echo present",
    f"git add {LOCKED_FILE}",
    "git status --short evals/locked",
    f"git commit -m 'exam' {LOCKED_FILE}",
    "uv run python -m portolan.bench chat-exam run --split test --max-usd 1",
    "cat evals/chat-exam/train.json",
    "cat evals/chat-exam/split.json",
    "grep -rn locked portolan/bench/chat_exam_cases.py",
]


@pytest.mark.parametrize("command", LOCKED_DENY)
def test_worktree_agents_cannot_read_the_locked_exam(command: str, worktree: Path) -> None:
    result = run_hook(bash(command, worktree))
    assert result.returncode == 2, f"not blocked: {command!r}\nstderr: {result.stderr}"
    assert "locked exam answers" in result.stderr
    assert "Exam test answers stay out of agents' reach" in result.stderr


@pytest.mark.parametrize("command", LOCKED_ALLOW)
def test_worktree_agents_can_list_the_locked_exam_and_run_it(command: str, worktree: Path) -> None:
    result = run_hook(bash(command, worktree))
    assert result.returncode == 0, f"blocked: {command!r}\nstderr: {result.stderr}"


@pytest.mark.parametrize("command", LOCKED_DENY[:4])
def test_main_checkout_can_read_the_locked_exam(command: str, main_checkout: Path) -> None:
    assert run_hook(bash(command, main_checkout)).returncode == 0, command


@pytest.mark.parametrize(
    "tool_input",
    [
        {"file_path": LOCKED_FILE},
        {"file_path": f"/Users/someone/portolan/.claude/worktrees/agent-x/{LOCKED_FILE}"},
        {"file_path": "/Users/someone/portolan/var/evals/locked/chat-exam/r/test.json.gz"},
        {"pattern": "renew", "path": "evals/locked"},
        {"pattern": "renew", "path": ".", "glob": "evals/locked/*.gz"},
    ],
    ids=lambda t: json.dumps(t, sort_keys=True),
)
def test_read_and_grep_tools_refuse_the_locked_exam_from_a_worktree(
    tool_input: dict, main_checkout: Path, worktree: Path
) -> None:
    tool = "Read" if "file_path" in tool_input else "Grep"
    blocked = run_hook({"tool_name": tool, "tool_input": tool_input, "cwd": str(worktree)})
    assert blocked.returncode == 2, blocked.stderr
    assert "locked exam answers" in blocked.stderr
    allowed = run_hook({"tool_name": tool, "tool_input": tool_input, "cwd": str(main_checkout)})
    assert allowed.returncode == 0, allowed.stderr


def test_the_train_half_stays_readable_from_a_worktree(worktree: Path) -> None:
    payload = {"tool_name": "Read", "tool_input": {"file_path": "evals/chat-exam/train.json"}}
    assert run_hook({**payload, "cwd": str(worktree)}).returncode == 0


def test_monitor_commands_get_the_bash_rules(main_checkout: Path, worktree: Path) -> None:
    assert run_hook(bash("tail -f .env", main_checkout, tool="Monitor")).returncode == 2
    assert run_hook(bash("railway logs --service web", worktree, tool="Monitor")).returncode == 2
    ok = "tail -f var/logs/run.log | grep --line-buffered -E 'ERROR|Traceback'"
    assert run_hook(bash(ok, main_checkout, tool="Monitor")).returncode == 0


def read(path: str) -> dict:
    return {
        "tool_name": "Read",
        "tool_input": {"file_path": path},
        "cwd": "/Users/someone/portolan",
    }


def grep_tool(**tool_input: str) -> dict:
    return {
        "tool_name": "Grep",
        "tool_input": {"pattern": "KEY", **tool_input},
        "cwd": "/Users/someone/portolan",
    }


@pytest.mark.parametrize(
    "payload",
    [
        read("/Users/someone/portolan/.env"),
        read("/Users/someone/portolan/.env.local"),
        read("/Users/someone/portolan/.envrc"),
        read("/Users/someone/portolan/docker/prod.env"),
        read("/Users/someone/portolan/var/railway/org-key.txt"),
        read("/Users/someone/portolan/var/railway/mcp-key.txt"),
        read("/Users/someone/.railway/config.json"),
        grep_tool(path="/Users/someone/portolan/.env"),
        grep_tool(path="/Users/someone/portolan/var/railway"),
        grep_tool(path="/Users/someone/portolan", glob=".env*"),
        grep_tool(path="/Users/someone/portolan", glob="**/.env"),
        grep_tool(glob="var/railway/*key*.txt"),
    ],
    ids=lambda p: f"{p['tool_name']}:{json.dumps(p['tool_input'], sort_keys=True)}",
)
def test_read_and_grep_tools_are_refused_on_secrets_files(payload: dict) -> None:
    result = run_hook(payload)
    assert result.returncode == 2, result.stderr
    assert f"no-secrets hook blocked this {payload['tool_name']} call" in result.stderr


@pytest.mark.parametrize(
    "payload",
    [
        read("/Users/someone/portolan/.env.example"),
        read("/Users/someone/portolan/portolan/config.py"),
        read("/Users/someone/portolan/var/railway/project.txt"),
        read("/Users/someone/portolan/01-spec/deploy.md"),
        read("/Users/someone/.docker/config.json"),
        read("/Users/someone/.aws/credentials"),
        read("/Users/someone/.pgpass"),
        grep_tool(path="/Users/someone/portolan/portolan"),
        grep_tool(path="/Users/someone/portolan", glob="*.py"),
        grep_tool(),
    ],
    ids=lambda p: f"{p['tool_name']}:{json.dumps(p['tool_input'], sort_keys=True)}",
)
def test_read_and_grep_tools_pass_on_ordinary_files(payload: dict) -> None:
    result = run_hook(payload)
    assert result.returncode == 0, result.stderr


def test_brace_bombs_cannot_crash_or_stall_it_open() -> None:
    # 1,100 brace groups once hit a RecursionError (exit 0); 22 took 20 s. The subprocess
    # timeout in run_hook is the bound.
    secret_read = "cat ." + "env"
    assert run_hook(bash(secret_read + " " + "{a,b}" * 1100)).returncode == 2
    assert run_hook(bash("{a,b}" * 1100 + "; " + secret_read)).returncode == 2
    assert run_hook(bash(secret_read + " " + "{a,b}" * 22)).returncode == 2
    assert run_hook(bash("echo " + "{a,b}" * 1100)).returncode == 0


def _load_hook_module():
    spec = importlib.util.spec_from_file_location("no_secrets_hook_under_test", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat .env", 2),
        ("echo $OPENROUTER_API_KEY", 2),
        ("railway up", 2),
        ("git push origin main", 2),
        ("ls -la portolan/", 0),
        ("uv run pytest -q tests/test_web.py", 0),
    ],
)
def test_an_internal_error_blocks_only_risky_text(
    command: str, expected: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    hook = _load_hook_module()

    def broken(*args, **kwargs):
        raise RecursionError("simulated")

    monkeypatch.setattr(hook, "decide", broken)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(bash(command))))
    assert hook.main() == expected
    if expected == 2:
        assert "could not be checked" in capsys.readouterr().err


def test_registered_command_says_so_and_allows_when_the_hook_file_is_missing(
    tmp_path: Path,
) -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    commands = {
        h["command"]
        for e in settings["hooks"]["PreToolUse"]
        for h in e["hooks"]
        if "no-secrets.py" in h["command"]
    }
    assert commands
    for command in commands:
        result = subprocess.run(
            ["sh", "-c", command],
            input=json.dumps(bash("printenv")),
            capture_output=True,
            text=True,
            timeout=30,
            env={"CLAUDE_PROJECT_DIR": str(tmp_path), "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
        )
        assert result.returncode == 0, result.stderr
        message = json.loads(result.stdout)["systemMessage"]
        assert "no-secrets.py is missing" in message


def test_settings_registers_the_hook_on_monitor_read_and_grep() -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    matched = set()
    for entry in settings["hooks"]["PreToolUse"]:
        if any("no-secrets.py" in h.get("command", "") for h in entry["hooks"]):
            matched.update(entry["matcher"].split("|"))
    assert {"Bash", "Monitor", "Read", "Grep"} <= matched


def test_settings_denies_reading_secrets_files() -> None:
    deny = json.loads(SETTINGS.read_text(encoding="utf-8"))["permissions"]["deny"]
    for rule in ("Read(**/.env)", "Read(**/.env.*)", "Read(**/var/railway/*key*.txt)"):
        assert rule in deny
    # the example is carved out after the rule it carves from (a `!` rule listed first does nothing)
    assert deny.index("Read(!**/.env.example)") > deny.index("Read(**/.env.*)")


# --------------------------------------------------------------------------- commit check

needs_gitleaks = pytest.mark.skipif(shutil.which("gitleaks") is None, reason="needs gitleaks")


def _git_env() -> dict:
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
        ["git", *args], cwd=repo, env=_git_env(), check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture()
def commit_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("hello\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


@needs_gitleaks
@pytest.mark.parametrize(
    "command", ["git add -A && git commit -m 'add notes'", "git commit -am 'x'"]
)
def test_a_commit_that_would_carry_a_secret_is_blocked(commit_repo: Path, command: str) -> None:
    key = "sk-or-" + "v1-" + secrets.token_hex(32)
    (commit_repo / "notes.md").write_text(f"the key was {key} yesterday\n")
    (commit_repo / "README.md").write_text(f"hello\n{key}\n")
    result = run_hook(bash(command, commit_repo))
    assert result.returncode == 2, result.stderr
    assert "portolan-openrouter-key" in result.stderr  # the repo's own .gitleaks.toml rule
    assert "notes.md:1" in result.stderr or "README.md:2" in result.stderr
    assert key not in result.stderr + result.stdout
    assert "Never commit secrets" in result.stderr
    # the scan used a throwaway index: nothing was staged for real
    assert _git(commit_repo, "diff", "--cached", "--name-only") == ""


@needs_gitleaks
def test_a_clean_commit_goes_ahead(commit_repo: Path) -> None:
    (commit_repo / "notes.md").write_text("nothing secret here\n")
    result = run_hook(bash("git add -A && git commit -m 'notes'", commit_repo))
    assert result.returncode == 0, result.stderr


@needs_gitleaks
def test_an_ignored_env_file_is_not_scanned(commit_repo: Path) -> None:
    (commit_repo / ".gitignore").write_text(".env\n")
    (commit_repo / ".env").write_text(
        "OPENROUTER_API_KEY=" + "sk-or-" + "v1-" + secrets.token_hex(32) + "\n"
    )
    result = run_hook(bash("git add -A && git commit -m 'ignore env'", commit_repo))
    assert result.returncode == 0, result.stderr
