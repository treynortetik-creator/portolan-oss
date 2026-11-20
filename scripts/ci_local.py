"""Run everything GitHub CI used to run, on this machine, and stamp the result.

    uv run python scripts/ci_local.py             # the pre-push gate
    uv run python scripts/ci_local.py --full      # never take the docs-only shortcut

GitHub Actions no longer runs on push (`.github/workflows/ci.yml` is `workflow_dispatch` only)
and Railway deploys `main` on push, so the stamp this writes, `var/ci/<HEAD sha>.json`, is the
gate: `.claude/hooks/push_gate.py` (the Claude Code hook and the git `pre-push` hook) refuses a
push without a passing stamp for HEAD's tree on a clean worktree. Commit first; the run refuses
to pass on uncommitted changes.

Steps, in order, stopping at the first failure:
  lockfile    `uv lock --check`
  lint        `ruff check .` and `ruff format --check .`
  truth-table `corpus/truth-table.yaml` parses
  secrets     gitleaks over the full git history with `.gitleaks.toml` (rule, file, line only)
  tests       the whole suite, with the environment CLAUDE.md gives (no .env, UTC, role test)
  docker      the old `docker` job: build docker/Dockerfile, run the suite in the container,
              migrate, and check the web role answers /login and /healthz with 200

Docs-only mode: when `scripts/affected_tests.py` says everything changed since the newest
passing stamp that is an ancestor of HEAD is documentation, the run is lint, lockfile,
truth-table, secrets and the repo-wide tests (the ones that read every tracked file).

Docker. The image is built from a `git archive HEAD` copy (committed files only, so the
throwaway `.env` the job writes never touches your real one) with `docker compose` under a
private project name, and the compose Postgres publishes no host port. Docker must be running;
without it the run fails at the start with "start it with `colima start`" and never skips the
step. Docs-only mode builds no image. Not covered: the image here is built natively for arm64
(Apple Silicon) while Railway builds amd64, so an architecture-specific wheel or binary problem
would only show up on Railway; `verify.py all` against the live site after each deploy is what
proves the deployed app.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import affected_tests  # noqa: E402  (same directory)

STAMP_DIR = Path("var") / "ci"


@dataclass
class StepResult:
    name: str
    result: str  # pass | fail | skipped
    seconds: float = 0.0
    detail: str = ""


@dataclass
class Stamp:
    sha: str
    tree: str
    clean: bool
    mode: str
    mode_reason: str
    result: str = "fail"
    started: str = ""
    steps: list = field(default_factory=list)
    tools: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- git helpers


def git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"ci_local: git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def dirty_files(root: Path) -> list[str]:
    return [line for line in git(root, "status", "--porcelain").splitlines() if line.strip()]


# --------------------------------------------------------------------------- mode


def mode_for_changes(root: Path, changed: list[str]) -> tuple[str, str]:
    """("docs-only" | "full", reason) for the paths changed since the last passing stamp."""
    graph = affected_tests.Graph(root, affected_tests.tracked_files(root))
    sel = affected_tests.select(graph, changed)
    if sel.full:
        return "full", "; ".join(sel.reasons[:3]) or "a change that needs the full suite"
    if sel.docs_only:
        return "docs-only", f"{len(changed)} changed path(s), all documentation"
    return "full", "code changed since the last passing stamp"


def last_pass_base(root: Path, head: str) -> str | None:
    """Newest passing stamp whose commit is an ancestor of HEAD (or HEAD itself)."""
    stamps = []
    for path in (root / STAMP_DIR).glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("result") == "pass" and data.get("clean") is True:
            stamps.append((path.stat().st_mtime, data.get("sha")))
    for _, sha in sorted(stamps, reverse=True):
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            continue
        anc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, head], cwd=root, capture_output=True
        )
        if anc.returncode == 0:
            return sha
    return None


def choose_mode(root: Path, head: str, force_full: bool) -> tuple[str, str]:
    if force_full:
        return "full", "--full"
    base = last_pass_base(root, head)
    if base is None:
        return "full", "no passing stamp for an ancestor of HEAD"
    changed = git(root, "diff", "--name-only", "--no-renames", base, head).split()
    if not changed:
        return "docs-only", f"no change since passing stamp {base[:10]}"
    mode, why = mode_for_changes(root, changed)
    return mode, f"since {base[:10]}: {why}"


# --------------------------------------------------------------------------- steps


def run(cmd: list[str], cwd: Path, env: dict | None = None, timeout: float | None = None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def tail(text: str, n: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def step_lockfile(root: Path) -> tuple[bool, str]:
    p = run(["uv", "lock", "--check"], root)
    return p.returncode == 0, tail(p.stdout + p.stderr, 5)


def step_lint(root: Path) -> tuple[bool, str]:
    for cmd in (["ruff", "check", "."], ["ruff", "format", "--check", "."]):
        p = run(["uv", "run", *cmd], root)
        if p.returncode != 0:
            return False, tail(p.stdout + p.stderr)
    return True, ""


def step_truth_table(root: Path) -> tuple[bool, str]:
    code = (
        "import yaml;"
        "d=yaml.safe_load(open('corpus/truth-table.yaml',encoding='utf-8'));"
        "assert isinstance(d,dict) and d,'empty or not a mapping';print(len(d))"
    )
    p = run(["uv", "run", "python", "-c", code], root)
    return p.returncode == 0, tail(p.stdout + p.stderr, 5)


def step_secrets(root: Path) -> tuple[bool, str]:
    gitleaks = shutil.which("gitleaks")
    if gitleaks is None:
        return False, "gitleaks is not installed (brew install gitleaks)"
    with tempfile.TemporaryDirectory(prefix="ci-local-") as td:
        report = os.path.join(td, "report.json")
        cmd = [
            gitleaks,
            "git",
            "--config",
            str(root / ".gitleaks.toml"),
            "--redact",
            "--no-banner",
            "--exit-code",
            "1",
            "--report-format",
            "json",
            "--report-path",
            report,
            str(root),
        ]
        p = run(cmd, root)
        if p.returncode == 0:
            return True, ""
        try:
            findings = json.loads(Path(report).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, f"gitleaks exited {p.returncode} and wrote no report"
    lines = [
        f"{f.get('RuleID')} {f.get('File')}:{f.get('StartLine')} (commit {str(f.get('Commit'))[:10]})"
        for f in findings
    ]
    return False, "\n".join(lines[:25]) + (
        f"\n... and {len(lines) - 25} more" if len(lines) > 25 else ""
    )


def step_tests(root: Path, mode: str) -> tuple[bool, str]:
    cmd = ["uv", "run", "pytest", "-q"]
    if mode == "docs-only":
        cmd += list(affected_tests.REPO_WIDE)
    p = run(cmd, root, env=affected_tests.pytest_env())
    return p.returncode == 0, tail(p.stdout + p.stderr, 40)


NO_DOCKER = (
    "Docker is not running or not installed; start it with `colima start`, then run ci_local again."
)


def _http_status(url: str) -> str:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310  (loopback URL)
            return str(resp.status)
    except Exception as exc:
        return (getattr(exc, "code", None) and str(exc.code)) or f"error: {type(exc).__name__}"


def docker_ready(root: Path) -> bool:
    return (
        shutil.which("docker") is not None
        and run(["docker", "info"], root, timeout=60).returncode == 0
    )


def step_docker(root: Path) -> tuple[bool, str]:
    """The `docker` job of the old workflow: build the image, run the suite in it, migrate, serve
    /login and /healthz. Built from a `git archive HEAD` copy so the throwaway `.env` the job
    writes never touches this checkout's real one, and only committed files reach the image."""
    if not docker_ready(root):
        return False, NO_DOCKER
    project = f"ci-local-{git(root, 'rev-parse', '--short=8', 'HEAD')}"
    with tempfile.TemporaryDirectory(prefix="ci-local-docker-") as td:
        tree = Path(td)
        archive = subprocess.Popen(["git", "archive", "HEAD"], cwd=root, stdout=subprocess.PIPE)
        untar = subprocess.run(["tar", "-x", "-C", str(tree)], stdin=archive.stdout)
        archive.stdout.close()
        if archive.wait() != 0 or untar.returncode != 0:
            return False, "git archive | tar failed"
        # The local Postgres and app usually hold 5432 and 8000, so publish neither.
        override = tree / "docker" / "docker-compose.ci-local.yml"
        override.write_text("services:\n  db:\n    ports: !reset []\n", encoding="utf-8")
        compose = [
            "docker", "compose", "-p", project,
            "-f", str(tree / "docker" / "docker-compose.yml"), "-f", str(override),
        ]  # fmt: skip
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        notes: list[str] = []

        def do(label: str, args: list[str]) -> str | None:
            began = time.monotonic()
            p = run([*compose, *args], tree, env=env)
            notes.append(f"{label}: {time.monotonic() - began:.0f}s")
            if p.returncode != 0:
                # stderr first: compose's progress lines would otherwise bury pytest's summary.
                return f"{label} failed:\n" + tail(p.stderr + p.stdout, 40)
            return None

        try:
            (tree / ".env").write_text("PORTOLAN_ORG_KEY=this-must-never-reach-the-built-image\n")
            failure = do("build", ["build", "app"])
            (tree / ".env").unlink()
            failure = failure or do(
                "no .env in image",
                ["run", "--rm", "--no-deps", "--entrypoint", "sh", "app", "-c",
                 "test ! -e /app/.env"],
            )  # fmt: skip
            failure = failure or do(
                "suite in container", ["run", "--rm", "-e", "PORTOLAN_ROLE=test", "app"]
            )
            failure = failure or do(
                "migrate", ["run", "--rm", "-e", "PORTOLAN_ROLE=migrate", "app"]
            )
            if not failure:
                web = run(
                    [*compose, "run", "-d", "-e", "PORTOLAN_ROLE=web",
                     "-e", "PORTOLAN_ORG_KEY=ci-docker-smoke-org-key",
                     "-e", "PORTOLAN_SESSION_SECRET=ci-docker-smoke-session-secret",
                     "-p", "127.0.0.1::8000", "app"],
                    tree, env=env,
                )  # fmt: skip
                cid = web.stdout.strip().splitlines()[-1] if web.returncode == 0 else ""
                if not cid:
                    failure = "starting the web role failed:\n" + tail(web.stdout + web.stderr, 20)
                else:
                    port = run(["docker", "port", cid, "8000"], tree).stdout.strip().split(":")[-1]
                    status = {}
                    for path in ("/login", "/healthz"):
                        code = ""
                        for _ in range(30):
                            code = _http_status(f"http://127.0.0.1:{port}{path}")
                            if code == "200":
                                break
                            time.sleep(1)
                        status[path] = code
                    notes.append("web: " + ", ".join(f"GET {k} -> {v}" for k, v in status.items()))
                    if set(status.values()) != {"200"}:
                        logs = run(["docker", "logs", "--tail", "30", cid], tree)
                        failure = "the web role did not answer 200:\n" + tail(
                            logs.stdout + logs.stderr, 30
                        )
        finally:
            run([*compose, "down", "-v", "--remove-orphans"], tree, env=env)
        return failure is None, failure or "\n".join(notes)


# --------------------------------------------------------------------------- run


def tool_versions(root: Path) -> dict:
    def first_line(cmd: list[str]) -> str:
        try:
            p = run(cmd, root, timeout=30)
            return (
                (p.stdout or p.stderr).strip().splitlines()[0]
                if (p.stdout or p.stderr).strip()
                else ""
            )
        except (OSError, subprocess.TimeoutExpired):
            return "unavailable"

    return {
        "python": sys.version.split()[0],
        "uv": first_line(["uv", "--version"]),
        "ruff": first_line(["uv", "run", "ruff", "--version"]),
        "pytest": first_line(["uv", "run", "pytest", "--version"]),
        "gitleaks": first_line(["gitleaks", "version"]),
        "git": first_line(["git", "--version"]),
    }


def write_stamp(root: Path, stamp: Stamp) -> Path:
    out = root / STAMP_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stamp.sha}.json"
    path.write_text(json.dumps(asdict(stamp), indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--full", action="store_true", help="never take the docs-only shortcut")
    ap.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="checkout to check (default: this one)"
    )
    args = ap.parse_args(argv)
    root = args.root.resolve()

    sha = git(root, "rev-parse", "HEAD")
    tree = git(root, "rev-parse", "HEAD^{tree}")
    dirty = dirty_files(root)
    mode, why = choose_mode(root, sha, args.full)
    stamp = Stamp(
        sha=sha,
        tree=tree,
        clean=not dirty,
        mode=mode,
        mode_reason=why,
        started=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    stamp.tools = tool_versions(root)

    steps = [
        ("lockfile", lambda: step_lockfile(root)),
        ("lint", lambda: step_lint(root)),
        ("truth-table", lambda: step_truth_table(root)),
        ("secrets", lambda: step_secrets(root)),
        ("tests", lambda: step_tests(root, mode)),
    ]
    if mode == "full":
        steps.append(("docker", lambda: step_docker(root)))

    if mode == "full" and not docker_ready(root):
        print(f"ci_local: FAIL. {NO_DOCKER} Nothing was run.")
        stamp.steps.append(asdict(StepResult("docker", "fail", 0.0, NO_DOCKER)))
        write_stamp(root, stamp)
        return 1

    print(f"ci_local: {sha[:10]} tree {tree[:10]}  mode {mode} ({why})")
    if dirty:
        print("ci_local: FAIL. Uncommitted changes; commit first (nothing was run):")
        print("\n".join("  " + line for line in dirty[:15]))
        stamp.steps.append(asdict(StepResult("clean-worktree", "fail", 0.0, "\n".join(dirty[:15]))))
        write_stamp(root, stamp)
        return 1

    failed = False
    for name, fn in steps:
        if failed:
            stamp.steps.append(asdict(StepResult(name, "skipped")))
            continue
        began = time.monotonic()
        try:
            ok, detail = fn()
        except Exception as exc:  # a crashed step is a failed step, never a pass
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        secs = round(time.monotonic() - began, 1)
        stamp.steps.append(
            asdict(
                StepResult(
                    name,
                    "pass" if ok else "fail",
                    secs,
                    detail if not ok or name == "docker" else "",
                )
            )
        )
        print(f"  {name:<12} {'pass' if ok else 'FAIL'}  {secs:>7.1f}s")
        if not ok:
            failed = True
            print("\n".join("    " + line for line in detail.splitlines()))

    if not failed and dirty_files(root):
        failed = True
        stamp.steps.append(
            asdict(StepResult("clean-worktree", "fail", 0.0, "the run left files changed"))
        )
        print(
            "  the run changed the worktree:\n"
            + "\n".join("    " + line for line in dirty_files(root)[:15])
        )
    stamp.clean = not dirty_files(root)
    stamp.result = "fail" if failed else "pass"
    path = write_stamp(root, stamp)
    print(f"ci_local: {stamp.result.upper()}  stamp {path.relative_to(root)}")
    return 0 if stamp.result == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
