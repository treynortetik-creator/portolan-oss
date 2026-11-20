"""Tests for `docker/**` and `.dockerignore` that don't need Docker itself
(review finding F4; `docker/README.md`: "Docker is not installed on the
primary dev Mac"). CI's `docker` job (`.github/workflows/ci.yml`) proves
the real build/run path; these prove the static contract every commit
can check without a Docker daemon: `.dockerignore` actually excludes
secrets and bulk, the entrypoint script exists and branches on every
documented `PORTOLAN_ROLE`, and the Dockerfile actually wires it in.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"


def _fake_bin(tmp_path: Path, name: str) -> Path:
    """A directory on PATH containing one fake executable `name` that just
    echoes `"{NAME}_ARGS: <its own argv>"` and exits 0 -- lets a test
    observe exactly what `entrypoint.sh` tried to `exec`, without that
    exec actually starting a server or running pytest."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    script = fake_bin / name
    script.write_text(f'#!/bin/sh\necho "{name.upper()}_ARGS: $*"\n')
    script.chmod(0o755)
    return fake_bin


def _non_comment_text(text: str) -> str:
    """`text` with every full-line `#`-comment dropped -- so a test
    asserting some old, now-fixed pattern is gone doesn't false-positive
    on that same literal string appearing in a comment explaining the
    fix (e.g. this file's own Dockerfile/entrypoint.sh docstrings quote
    `|| true` and `ghcr.io/astral-sh/uv:latest` verbatim while explaining
    why each is gone)."""
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def _run_entrypoint(
    tmp_path: Path, *, role: str | None, intercept: str, cmd_args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    fake_bin = _fake_bin(tmp_path, intercept)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    if role is None:
        env.pop("PORTOLAN_ROLE", None)
    else:
        env["PORTOLAN_ROLE"] = role
    argv = ["sh", str(ENTRYPOINT), *(cmd_args or [])]
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=10)


def test_dockerignore_exists_and_excludes_secrets_and_bulk() -> None:
    assert DOCKERIGNORE.exists(), ".dockerignore is missing (review finding F4)"
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}

    # The exact defect the review proved: .env baked into a layer pushed to ECR.
    assert ".env" in lines
    assert any(line.startswith(".env") for line in lines)
    assert ".env.example" not in lines  # the template should still be copyable

    for required in (".venv/", "var/", "corpus/out-large/", "00-context/", ".git/", ".remember/"):
        assert required in lines, f"{required!r} missing from .dockerignore"


def test_dockerignore_does_not_exclude_application_code() -> None:
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    lines = [
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    # A pattern like "portolan/" or "tests/" here would silently break the
    # image -- guard against that class of mistake.
    for line in lines:
        assert not line.startswith("portolan"), f"would exclude application code: {line!r}"
        assert not line.startswith("tests"), f"would exclude the test suite: {line!r}"
        assert line not in ("pyproject.toml", "uv.lock", "docker/")


def test_entrypoint_script_exists_and_is_executable() -> None:
    assert ENTRYPOINT.exists(), "docker/entrypoint.sh is missing (review finding F4)"
    mode = ENTRYPOINT.stat().st_mode
    assert mode & stat.S_IXUSR, "docker/entrypoint.sh is not executable"


def test_entrypoint_script_branches_on_every_documented_role() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "PORTOLAN_ROLE" in text
    for role in ("web", "worker", "migrate", "test"):
        assert f"{role})" in text, f"entrypoint.sh has no {role!r} case"
    # web must actually serve the app, not just print something.
    assert "portolan.web" in text
    # migrate must actually touch the (bug hunt 2026-09-18 P0 finding 2)
    # backup-then-apply migration path, not the bare, backup-less
    # apply_migrations() call this used to be an inline call to.
    assert "portolan.db migrate" in text


def test_dockerfile_wires_in_the_entrypoint() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "ENTRYPOINT" in text
    assert "entrypoint.sh" in text
    assert "EXPOSE 8000" in text
    # The old defect: CMD alone ran the test suite with no entrypoint at
    # all binding anything. A CMD-only image (no ENTRYPOINT) is exactly
    # what F4 found -- guard the fix stays in place.
    assert "ENTRYPOINT [" in text


# --------------------------------------------------------------------
# Bug hunt 2026-09-18 (ops review, P1 finding 4): an unset/dropped
# PORTOLAN_ROLE used to fall through to the image's CMD (the test
# suite). These actually run docker/entrypoint.sh under `sh`, with a fake
# `python`/`uv` on PATH intercepting the `exec` call, so they prove what
# the script really does -- not just what its source text says.
# --------------------------------------------------------------------


def test_entrypoint_defaults_an_unset_role_to_web_not_the_cmd_args(tmp_path: Path) -> None:
    result = _run_entrypoint(
        tmp_path, role=None, intercept="python", cmd_args=["uv", "run", "pytest", "-q"]
    )
    assert result.returncode == 0, result.stderr
    assert "PYTHON_ARGS: -m portolan.web" in result.stdout
    # The old defect, proven absent: the CMD args ("uv run pytest -q")
    # never got anywhere near being executed.
    assert "pytest" not in result.stdout


def test_entrypoint_empty_string_role_also_defaults_to_web(tmp_path: Path) -> None:
    result = _run_entrypoint(tmp_path, role="", intercept="python")
    assert result.returncode == 0, result.stderr
    assert "PYTHON_ARGS: -m portolan.web" in result.stdout


def test_entrypoint_migrate_role_execs_the_backup_taking_cli(tmp_path: Path) -> None:
    result = _run_entrypoint(tmp_path, role="migrate", intercept="python")
    assert result.returncode == 0, result.stderr
    assert "PYTHON_ARGS: -m portolan.db migrate" in result.stdout


def test_entrypoint_test_role_execs_pytest(tmp_path: Path) -> None:
    result = _run_entrypoint(tmp_path, role="test", intercept="uv")
    assert result.returncode == 0, result.stderr
    assert "UV_ARGS: run pytest -q" in result.stdout


def test_entrypoint_exits_non_zero_with_a_clear_message_on_an_unknown_role(tmp_path: Path) -> None:
    result = _run_entrypoint(tmp_path, role="bogus-role", intercept="python")
    assert result.returncode != 0
    assert "unknown PORTOLAN_ROLE" in result.stderr
    assert "bogus-role" in result.stderr


def test_entrypoint_worker_loop_no_longer_swallows_a_failed_cycle() -> None:
    """Bug hunt 2026-09-18, P2 finding 15: `|| true` made a failing worker
    cycle indistinguishable from a succeeding one in the logs."""
    text = _non_comment_text(ENTRYPOINT.read_text(encoding="utf-8"))
    assert "|| true" not in text
    assert "cycle failed" in text


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, P1 finding 9: the base image and the uv installer
# image were both floating tags -- a release between CI's build and
# Railway's own build of the same commit could ship two different images
# from one green check.
# --------------------------------------------------------------------


def test_dockerfile_pins_the_python_base_image_by_digest() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    match = re.search(r"^FROM\s+(\S+)", text, re.MULTILINE)
    assert match, "no FROM line found"
    assert "@sha256:" in match.group(1), f"base image is not pinned by digest: {match.group(1)!r}"
    assert match.group(1) != "python:3.13-slim@sha256:", "digest placeholder was never filled in"


def test_dockerfile_pins_the_uv_installer_image_by_tag_and_digest() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    non_comment = _non_comment_text(text)
    match = re.search(r"COPY --from=(\S+)", non_comment)
    assert match, "no `COPY --from=...uv...` line found"
    ref = match.group(1)
    assert "ghcr.io/astral-sh/uv:latest" not in non_comment, (
        "uv installer image is still the floating tag"
    )
    assert ":0.11.3@sha256:" in ref, (
        f"uv installer image is not pinned to a version+digest: {ref!r}"
    )


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, P0 finding 2: pg_dump/pg_restore must exist in the
# image for the pre-migration backup / restore CLI to work at all.
# --------------------------------------------------------------------


def test_dockerfile_installs_postgresql_client_matching_the_deployed_major_version() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    # docker-compose.yml / CI both run pgvector/pgvector:pg17.
    assert "postgresql-client-17" in text


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, P2 finding 12: no HEALTHCHECK.
# --------------------------------------------------------------------


def test_dockerfile_has_a_healthcheck_pointed_at_healthz() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "HEALTHCHECK" in text
    assert "/healthz" in text


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, P2 finding 11: .claude/, infra/, scripts/ were
# missing from .dockerignore. 01-spec/ and 04-brand/ are deliberately
# NOT added -- both are read at runtime (portolan/web/routes/
# how_it_works.py, portolan/web/templating.py), so excluding them would
# ship a broken image (verified locally; see the "Fix status" table in
# 01-spec/reviews/bughunt-2026-09-18-ops.md).
# --------------------------------------------------------------------


def test_dockerignore_excludes_dev_only_directories_added_2026_09_18() -> None:
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}
    for required in (".claude/", "infra/", "scripts/"):
        assert required in lines, f"{required!r} missing from .dockerignore"


def test_dockerignore_still_ships_the_runtime_required_directories() -> None:
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    lines = {
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    }
    for runtime_required in ("01-spec/", "04-brand/"):
        assert runtime_required not in lines, (
            f"{runtime_required!r} must not be dockerignored -- it is read at runtime "
            "(portolan/web/routes/how_it_works.py, portolan/web/templating.py)"
        )


def _run_entrypoint_seeing_role(tmp_path: Path, *, role: str | None) -> str:
    """Runs the entrypoint with a fake `python` that prints the
    `PORTOLAN_ROLE` it inherited: `portolan.web.__main__` binds all
    interfaces only when that variable is set."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "python"
    fake_python.write_text('#!/bin/sh\necho "ROLE_SEEN: ${PORTOLAN_ROLE-unset}"\n')
    fake_python.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    if role is None:
        env.pop("PORTOLAN_ROLE", None)
    else:
        env["PORTOLAN_ROLE"] = role
    result = subprocess.run(
        ["sh", str(ENTRYPOINT)], capture_output=True, text=True, env=env, timeout=10
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_entrypoint_exports_the_defaulted_web_role_to_the_app(tmp_path: Path) -> None:
    assert "ROLE_SEEN: web" in _run_entrypoint_seeing_role(tmp_path, role=None)
    assert "ROLE_SEEN: web" in _run_entrypoint_seeing_role(tmp_path, role="")
