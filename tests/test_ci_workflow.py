"""Static checks on `.github/workflows/ci.yml` (bug hunt 2026-09-18, P2
finding 14: unfrozen lock in the `test` job, no concurrency guard). Not a
substitute for actually running the workflow -- just proves the specific
lines the finding named are the ones actually in the file.

CI-vs-docker fix (2026-09-19): `.dockerignore` deliberately excludes
`.git/` and `.github/` from the image (never needed at runtime, and the
former would bake the whole repo history into a layer -- see its own
comment) -- so the `docker` CI job, which runs this exact suite inside
the built container (`PORTOLAN_ROLE=test`), has no `.github/workflows/
ci.yml` to read at all; `CI_YML.read_text()` raised a bare
`FileNotFoundError` there (`/app/.github/workflows/ci.yml`) that had
nothing to do with the checks this file actually wants to make. Skipping
the whole module when the file is absent -- rather than trying to locate
some OTHER checkout via `git rev-parse` -- is the right call here
specifically because there both is and can be no such checkout inside
the container image; these are static checks on this repo's own workflow
file, only meaningful from a real git checkout that still has it.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from tests._paths import require_repo_path

REPO_ROOT = Path(__file__).resolve().parent.parent
# .dockerignore excludes .github/ from the image, so this static read of the
# workflow file only applies inside a real git checkout.
CI_YML = require_repo_path(".github", "workflows", "ci.yml")


def test_test_job_installs_dependencies_frozen() -> None:
    workflow = yaml.safe_load(CI_YML.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    install_steps = [s for s in steps if s.get("name") == "Install dependencies"]
    assert install_steps, "no 'Install dependencies' step found in the test job"
    assert install_steps[0]["run"].strip() == "uv sync --frozen"


def test_workflow_has_a_concurrency_group_keyed_on_the_ref() -> None:
    workflow = yaml.safe_load(CI_YML.read_text(encoding="utf-8"))
    concurrency = workflow.get("concurrency")
    assert concurrency, "no top-level `concurrency:` block"
    assert concurrency["cancel-in-progress"] is True
    assert "github.ref" in concurrency["group"]


def test_test_job_installs_a_matching_postgresql_client_before_testing() -> None:
    """pg_dump/pg_restore server-version-mismatch fix: `portolan.db`'s
    pre-migration backup path shells out to client tools matching the
    connected server's major version (`portolan.db._resolve_pg_binary`).
    The `test` job's Postgres service is `pgvector/pgvector:pg17`
    (asserted below too), so a matching `postgresql-client-17` must be
    installed, and before the `Test` step runs it."""
    workflow = yaml.safe_load(CI_YML.read_text(encoding="utf-8"))
    test_job = workflow["jobs"]["test"]
    assert test_job["services"]["postgres"]["image"] == "pgvector/pgvector:pg17"

    steps = test_job["steps"]
    names = [s.get("name", "") for s in steps]
    install_idx = next((i for i, name in enumerate(names) if "postgresql-client-17" in name), None)
    assert install_idx is not None, "no step installs postgresql-client-17"
    assert "postgresql-client-17" in steps[install_idx]["run"]

    test_idx = next(i for i, s in enumerate(steps) if s.get("name") == "Test")
    assert install_idx < test_idx, "postgresql-client-17 must be installed before the Test step"


def test_docker_job_test_suite_step_sets_an_explicit_role() -> None:
    """Bug hunt 2026-09-18, P1 finding 4: running the suite in-container
    used to rely on an UNSET PORTOLAN_ROLE falling through to the image's
    CMD -- that fallthrough is gone, so this step must ask for it
    explicitly instead."""
    text = CI_YML.read_text(encoding="utf-8")
    assert "PORTOLAN_ROLE=test" in text


def _workflow() -> dict:
    return yaml.safe_load(CI_YML.read_text(encoding="utf-8"))


def _step_index(steps: list, predicate) -> int:
    return next(i for i, s in enumerate(steps) if predicate(s))


def test_every_job_has_a_timeout() -> None:
    # A hung job would otherwise hold Railway's Wait-for-CI for GitHub's 6-hour default.
    for name, job in _workflow()["jobs"].items():
        minutes = job.get("timeout-minutes")
        assert isinstance(minutes, int) and 0 < minutes <= 60, (name, minutes)


def test_every_action_is_pinned_to_a_commit_sha() -> None:
    uses = [
        step["uses"]
        for job in _workflow()["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    ]
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


def test_lint_step_checks_formatting_too() -> None:
    steps = _workflow()["jobs"]["test"]["steps"]
    lint = steps[_step_index(steps, lambda s: s.get("name") == "Lint")]["run"]
    assert "uv run ruff check ." in lint
    assert "uv run ruff format --check ." in lint


def test_lockfile_is_checked_before_the_frozen_install() -> None:
    steps = _workflow()["jobs"]["test"]["steps"]
    check = _step_index(steps, lambda s: s.get("run", "").strip() == "uv lock --check")
    install = _step_index(steps, lambda s: s.get("name") == "Install dependencies")
    assert check < install


def test_docker_job_curls_healthz_after_starting_the_web_role() -> None:
    steps = _workflow()["jobs"]["docker"]["steps"]
    web = _step_index(steps, lambda s: "PORTOLAN_ROLE=web" in s.get("run", ""))
    health = _step_index(steps, lambda s: "/healthz" in s.get("run", ""))
    assert web < health
    assert 'test "$status" = "200"' in steps[health]["run"]


def test_test_job_installs_the_same_pinned_gitleaks_before_testing() -> None:
    jobs = _workflow()["jobs"]
    for key in ("GITLEAKS_VERSION", "GITLEAKS_SHA256"):
        assert jobs["test"]["env"][key] == jobs["secrets"]["env"][key]
    steps = jobs["test"]["steps"]
    install = _step_index(steps, lambda s: "Install gitleaks" in s.get("name", ""))
    assert "sha256sum -c" in steps[install]["run"]
    assert install < _step_index(steps, lambda s: s.get("name") == "Test")
