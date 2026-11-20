"""The local push gate: a push is allowed only when `scripts/ci_local.py` passed on exactly the
commit being pushed. Shared by the Claude Code hook (`no-secrets.py`, on `git push`) and the git
`pre-push` hook (`scripts/git-hooks/pre-push`), so an agent and a human are held to one rule.

GitHub CI no longer runs on push and Railway deploys `main` on push, so this check is the only
thing between a push and a live deploy.

A push is allowed when `<repo>/var/ci/<commit sha>.json` says pass, its tree hash equals the
commit's tree, and the worktree is clean. The stamp is a plain file, not a signature: this
stops the accident (pushing what was never checked, or changed after the check), not someone
who writes a stamp by hand.

    python3 push_gate.py [--sha REV]     exit 0 = allowed, 1 = refused (reasons on stderr)

Runs on the system python3 (3.9 on macOS): no 3.10+ syntax.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

STAMP_DIR = os.path.join("var", "ci")
ADVICE = "run `uv run python scripts/ci_local.py` from a clean worktree, then push again."


def _git(cwd, *args):
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)
    return proc.returncode, proc.stdout.strip()


def stamp_path(top, sha):
    return os.path.join(top, STAMP_DIR, sha + ".json")


def problems(repo_dir, rev="HEAD"):
    """Reasons the commit `rev` in repo_dir may not be pushed; [] when it may.
    Not inside a git checkout: [] (there is nothing to push)."""
    if not os.path.isdir(repo_dir):
        return []
    code, top = _git(repo_dir, "rev-parse", "--show-toplevel")
    if code != 0:
        return []
    code, sha = _git(top, "rev-parse", "--verify", rev + "^{commit}")
    if code != 0:
        return ["cannot resolve %s to a commit" % rev]
    _, tree = _git(top, "rev-parse", sha + "^{tree}")
    path = stamp_path(top, sha)
    try:
        with open(path, encoding="utf-8") as fh:
            stamp = json.load(fh)
        if not isinstance(stamp, dict):
            raise ValueError("not an object")
    except FileNotFoundError:
        return [
            "no CI stamp for commit %s (%s is missing)" % (sha[:10], os.path.relpath(path, top))
        ]
    except (OSError, ValueError):
        return ["the CI stamp for commit %s is unreadable" % sha[:10]]
    out = []
    steps = stamp.get("steps")
    if (
        stamp.get("result") != "pass"
        or not isinstance(steps, list)
        or not steps
        or any(not isinstance(s, dict) or s.get("result") != "pass" for s in steps)
    ):
        out.append("the CI stamp for commit %s does not say pass" % sha[:10])
    if stamp.get("sha") != sha:
        out.append("the CI stamp names a different commit")
    if stamp.get("tree") != tree:
        out.append("the CI stamp is for a different tree (the code changed after CI ran)")
    if stamp.get("clean") is not True:
        out.append("CI ran on a worktree with uncommitted changes")
    code, status = _git(top, "status", "--porcelain")
    if code != 0 or status:
        out.append("the worktree has uncommitted changes")
    return out


def message(reasons):
    return "push refused by the local CI gate: " + "; ".join(reasons) + "."


def main(argv):
    rev = "HEAD"
    if "--sha" in argv:
        rev = argv[argv.index("--sha") + 1]
    reasons = problems(os.getcwd(), rev)
    if not reasons:
        return 0
    sys.stderr.write(message(reasons) + "\nInstead: " + ADVICE + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
