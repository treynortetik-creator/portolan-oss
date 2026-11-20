"""The suite's own guardrails (`tests/guards.py`; guardrails plan 2026-09-22, ranks 1 and 3).

In-process tests call the guarded functions directly and then drain the recorded
violation with `guards.take_violations()`, since a test that catches a guard exception is
otherwise failed on purpose. End-to-end tests start a separate pytest in a subprocess with a
fake remote DSN; nothing ever connects to it (the refusal happens before collection, and the
connect guard fires before libpq is called)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import psycopg
import pytest

from tests import guards

REPO_ROOT = Path(__file__).resolve().parent.parent
VAR = REPO_ROOT / "var"
PROBE = "guard-probe-must-never-exist"
FAKE_PASSWORD = "hunter2-not-a-real-password"
REMOTE_HOST = "db.example.invalid"
CI_ENV = {"GITHUB_ACTIONS": "true"}  # what GitHub sets on every Actions job


@contextlib.contextmanager
def refused(match: str | None = None):
    """`pytest.raises(GuardViolation)`, then drain the recorded violation so the
    swallowed-violation check in `pytest_runtest_makereport` does not fail this test."""
    with pytest.raises(guards.GuardViolation, match=match) as info:
        yield info
    guards.take_violations()


# --------------------------------------------------------------------------- 1. database


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://localhost/portolan_test",
        "postgresql://localhost/portolan_test_1234",
        "postgresql://portolan:portolan@localhost:5432/portolan_test",  # CI's test job
        "postgresql://127.0.0.1/portolan_test_x",
        "postgresql://[::1]:5432/portolan_test",
        "postgresql:///portolan_test",  # default unix socket
        "postgresql://%2Ftmp/portolan_test",
        "host=/tmp dbname=portolan_test_pipeline",
    ],
)
def test_local_test_databases_pass(dsn: str) -> None:
    assert guards.database_refusal(dsn, CI_ENV, container=False, allow_maintenance=False) is None


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://localhost/portolan_test",
        "postgresql://portolan:portolan@localhost:5432/portolan_test",  # CI's test job
        "postgresql:///portolan_test",
        "host=/tmp dbname=portolan_test",
    ],
)
def test_bare_test_database_name_needs_the_ci_marker(dsn: str) -> None:
    """`portolan_test` was the dev graph's name, so a `.env`-exported run would pass the
    prefix rule and wipe it. Outside CI only suffixed names pass."""
    for environ in ({}, {"GITHUB_ACTIONS": "false"}, {"CI": "true"}):
        reason = guards.database_refusal(dsn, environ, container=False, allow_maintenance=False)
        assert reason is not None and "old dev-graph name" in reason
    assert guards.database_refusal(dsn, CI_ENV, container=False, allow_maintenance=False) is None
    suffixed = dsn.replace("portolan_test", "portolan_test_4242")
    assert guards.database_refusal(suffixed, {}, container=False, allow_maintenance=False) is None


@pytest.mark.parametrize(
    ("dsn", "environ", "fragment"),
    [
        (f"postgresql://u:{FAKE_PASSWORD}@{REMOTE_HOST}:5432/portolan_test", {}, "not local"),
        ("postgresql://localhost/portolan", {}, "'portolan'"),
        ("postgresql://localhost/railway", {}, "'railway'"),
        ("postgresql://localhost/my_portolan_test", {}, "starts with"),
        (f"postgresql:///portolan_test?host={REMOTE_HOST}", {}, "not local"),
        ("postgresql://localhost/portolan_test?dbname=portolan", {}, "'portolan'"),
        ("postgresql://localhost,remote.example.invalid/portolan_test", {}, "not local"),
        ("postgresql://localhost/portolan_test?hostaddr=203.0.113.5", {}, "hostaddr"),
        ("postgresql://localhost/portolan_test", {"PGHOSTADDR": "203.0.113.5"}, "hostaddr"),
        ("postgresql:///portolan_test", {"PGHOST": REMOTE_HOST}, "not local"),
        ("postgresql://localhost", {"PGDATABASE": "portolan"}, "'portolan'"),
        ("postgresql://localhost/portolan_test", {"PGSERVICE": "prod"}, "service"),
        ("postgresql://db:5432/portolan_test", {}, "not local"),  # `db` only inside a container
        ("not a dsn at all ===", {}, "does not parse"),
    ],
)
def test_non_test_or_remote_databases_are_refused(dsn: str, environ: dict, fragment: str) -> None:
    reason = guards.database_refusal(dsn, environ, container=False, allow_maintenance=False)
    assert reason is not None and fragment in reason
    assert FAKE_PASSWORD not in reason and REMOTE_HOST not in reason


def test_compose_service_host_passes_only_in_a_container() -> None:
    dsn = "postgresql://portolan:portolan@db:5432/portolan_test_compose"  # docker-compose.yml
    assert guards.database_refusal(dsn, {}, container=True, allow_maintenance=False) is None
    # psycopg resolves `db` to the bridge address before libpq connects
    resolved = "host=db hostaddr=172.18.0.2 dbname=portolan_test_compose user=portolan"
    assert guards.database_refusal(resolved, {}, container=True, allow_maintenance=True) is None
    assert guards.database_refusal(resolved, {}, container=False, allow_maintenance=True)


def test_maintenance_database_only_per_connection() -> None:
    dsn = "postgresql://localhost/postgres"
    assert guards.database_refusal(dsn, {}, container=False, allow_maintenance=True) is None
    assert guards.database_refusal(dsn, {}, container=False, allow_maintenance=False)


def test_session_refuses_railway_variables_by_name_only() -> None:
    reason = guards.session_refusal(
        {"RAILWAY_ENVIRONMENT": "production", "RAILWAY_TCP_PROXY_DOMAIN": REMOTE_HOST},
        container=False,
    )
    assert reason and "RAILWAY_ENVIRONMENT" in reason and "railway run" in reason
    assert REMOTE_HOST not in reason and "production" not in reason


def test_session_checks_the_default_database_against_pg_variables() -> None:
    assert guards.session_refusal({}, container=False) is None
    assert guards.session_refusal({"PGHOSTADDR": "203.0.113.5"}, container=False)
    assert guards.session_refusal({"PGSERVICE": "live"}, container=False)


def test_connect_guard_refuses_a_non_test_database_before_libpq() -> None:
    with refused("names database 'portolan_prod'"):
        psycopg.connect("postgresql://localhost/portolan_prod", connect_timeout=1)


def test_connect_guard_refuses_the_bare_dev_name_outside_ci(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with refused("old dev-graph name"):
        psycopg.connect("postgresql://localhost/portolan_test", connect_timeout=1)


def test_connect_guard_refuses_a_remote_host_before_libpq() -> None:
    dsn = f"postgresql://u:{FAKE_PASSWORD}@203.0.113.5:5432/portolan_test"
    with refused() as info:
        psycopg.connect(dsn, connect_timeout=1)
    assert "not local" in str(info.value) and FAKE_PASSWORD not in str(info.value)
    assert "test_connect_guard_refuses_a_remote_host_before_libpq" in str(info.value)


def test_connect_guard_covers_keyword_arguments_and_async() -> None:
    with refused():
        psycopg.connect(
            "postgresql://localhost/portolan_test", dbname="portolan", connect_timeout=1
        )

    async def go() -> None:
        await psycopg.AsyncConnection.connect(
            "postgresql://localhost/portolan_live", connect_timeout=1
        )

    with refused():
        asyncio.run(go())


def test_connect_guard_lets_the_test_database_through(db) -> None:
    with db.cursor() as cur:
        cur.execute("SELECT current_database()")
        assert cur.fetchone()[0].startswith("portolan_test")


# --------------------------------------------------------------------------- 2. network


def test_remote_socket_connect_is_refused_and_names_the_test() -> None:
    with refused() as info:
        socket.create_connection(("203.0.113.5", 443), timeout=1)
    message = str(info.value)
    assert "203.0.113.5:443" in message
    assert "test_remote_socket_connect_is_refused_and_names_the_test" in message


def test_httpx_to_a_remote_host_is_refused() -> None:
    with refused():
        httpx.get("http://203.0.113.5/api/v1/key", timeout=1)


def test_loopback_and_unix_sockets_still_work() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        client = socket.create_connection(server.getsockname(), timeout=2)
        client.close()
    finally:
        server.close()

    # /tmp, not tmp_path: macOS caps a unix socket path at 104 bytes
    sock_dir = tempfile.mkdtemp(prefix="g", dir="/tmp")
    path = os.path.join(sock_dir, "s")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(path)
        listener.listen(1)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(path)
        client.close()
    finally:
        listener.close()
        shutil.rmtree(sock_dir, ignore_errors=True)


@pytest.mark.parametrize(
    ("address", "local"),
    [
        (("127.0.0.1", 80), True),
        (("::1", 80, 0, 0), True),
        (("localhost", 80), True),
        (("::ffff:127.0.0.1", 80, 0, 0), True),
        (("10.0.0.5", 5432), False),
        (("openrouter.ai", 443), False),
    ],
)
def test_address_classification(address, local: bool) -> None:
    family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
    assert guards.address_is_local(family, address) is local


# --------------------------------------------------------------------------- 3. paid keys


def test_paid_model_keys_are_absent() -> None:
    for name in guards.PAID_MODEL_KEYS:
        assert name not in os.environ


def test_dotenv_reads_hide_paid_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from corpus.generator.prose import _env_or_dotenv
    from portolan.config import Settings

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY=fake-or-value\nexport TYPESAFE_API_KEY=fake-ts\n"
        "ANTHROPIC_API_KEY = fake-an\nPORTOLAN_TENANT=acme\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(guards.STATE, "dotenv_paths", {os.path.realpath(env_file)})

    text = env_file.read_text(encoding="utf-8")
    assert "fake" not in text and "PORTOLAN_TENANT=acme" in text
    with open(env_file, "rb") as fh:
        assert b"fake" not in fh.read()
    assert _env_or_dotenv(tmp_path, "OPENROUTER_API_KEY") is None
    settings = Settings(_env_file=env_file)
    assert settings.openrouter_api_key is None and settings.anthropic_api_key is None
    assert settings.tenant == "acme"

    with guards.dotenv_keys_visible():
        assert _env_or_dotenv(tmp_path, "OPENROUTER_API_KEY") == "fake-or-value"


# --------------------------------------------------------------------------- 4. var/


def _assert_blocked(fn, *args, **kwargs) -> None:
    with refused("wrote under the repo's var/"):
        fn(*args, **kwargs)


def test_writes_under_repo_var_are_refused(tmp_path: Path) -> None:
    target = VAR / "data" / PROBE
    src = tmp_path / "src.txt"
    src.write_text("x", encoding="utf-8")

    _assert_blocked(open, target, "w")
    _assert_blocked(open, target, "a")
    _assert_blocked(open, target, mode="xb")
    _assert_blocked(target.write_text, "fake record")
    _assert_blocked(target.write_bytes, b"fake record")
    _assert_blocked(target.touch)
    _assert_blocked(target.mkdir, parents=True)
    _assert_blocked(os.makedirs, VAR / PROBE, exist_ok=True)
    _assert_blocked(os.open, target, os.O_CREAT | os.O_WRONLY)
    _assert_blocked(shutil.copy, src, target)
    _assert_blocked(os.replace, src, target)
    _assert_blocked(os.rename, src, target)
    _assert_blocked(os.remove, target)
    _assert_blocked(os.unlink, target)
    _assert_blocked(shutil.rmtree, VAR / PROBE)
    _assert_blocked(tempfile.mkstemp, dir=VAR)
    _assert_blocked(logging.FileHandler, VAR / "logs" / PROBE)
    _assert_blocked(open, "var/" + PROBE, "w")  # relative to the repo-root cwd pytest runs in
    assert not target.exists() and not (VAR / PROBE).exists()
    assert src.exists()  # os.replace / os.rename were refused before moving anything


def test_reads_and_tmp_path_writes_are_untouched(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open(VAR / "data" / PROBE)  # noqa: SIM115 -- a read goes straight through, and raises
    (tmp_path / "var" / "data").mkdir(parents=True)  # a var/ that is not the repo's
    (tmp_path / "var" / "data" / "ok.json").write_text("{}", encoding="utf-8")


def test_main_checkout_root_from_a_worktree() -> None:
    wt = Path("/x/living-knowledge-graph/.claude/worktrees/agent-1")
    assert guards.main_checkout_root(wt) == Path("/x/living-knowledge-graph")
    assert guards.main_checkout_root(Path("/x/living-knowledge-graph")) is None


# --------------------------------------------------------------------------- end to end


def _clean_env(**extra: str) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("RAILWAY_", "PG"))
        and k not in ("DATABASE_URL", *guards.PAID_MODEL_KEYS)
    }
    env.update(extra)
    return env


def _pytest(args: list[str], env: dict, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


COLLECT_ONE = ["--collect-only", "-q", "tests/test_util_redact.py"]


@pytest.mark.parametrize(
    ("extra_env", "extra_args", "fragment"),
    [
        (
            {"DATABASE_URL": f"postgresql://u:{FAKE_PASSWORD}@{REMOTE_HOST}:5432/portolan_test"},
            [],
            "not local",
        ),
        ({"DATABASE_URL": "postgresql://localhost/portolan"}, [], "'portolan'"),
        (
            {
                "DATABASE_URL": "postgresql://localhost/portolan_test",
                "RAILWAY_ENVIRONMENT": "production",
            },
            [],
            "RAILWAY_",
        ),
        ({"PGHOSTADDR": "203.0.113.5"}, [], "hostaddr"),
        (
            {"DATABASE_URL": f"postgresql://u:{FAKE_PASSWORD}@{REMOTE_HOST}/portolan_test"},
            ["--noconftest"],
            "not local",
        ),
        (
            {"DATABASE_URL": f"postgresql://u:{FAKE_PASSWORD}@{REMOTE_HOST}/portolan_test"},
            ["-c", os.devnull, "--rootdir", str(REPO_ROOT)],
            "not local",
        ),
        ({}, ["-p", "no:tests.guards"], "disabled"),
    ],
)
def test_suite_refuses_to_start(extra_env: dict, extra_args: list[str], fragment: str) -> None:
    result = _pytest([*extra_args, *COLLECT_ONE], _clean_env(**extra_env))
    output = result.stdout + result.stderr
    assert result.returncode == 4, output
    assert "Refusing to run the test suite" in output and fragment in output
    assert FAKE_PASSWORD not in output and REMOTE_HOST not in output


def test_suite_refuses_the_bare_dev_name_outside_ci_and_starts_in_ci() -> None:
    env = _clean_env(DATABASE_URL="postgresql://localhost/portolan_test")
    env.pop("GITHUB_ACTIONS", None)
    refused_run = _pytest(COLLECT_ONE, env)
    output = refused_run.stdout + refused_run.stderr
    assert refused_run.returncode == 4, output
    assert "Refusing to run the test suite" in output and "old dev-graph name" in output

    ci_run = _pytest(COLLECT_ONE, {**env, **CI_ENV})
    assert ci_run.returncode == 0, ci_run.stdout + ci_run.stderr


def test_suite_starts_against_a_local_test_database() -> None:
    result = _pytest(
        COLLECT_ONE, _clean_env(DATABASE_URL="postgresql://localhost/portolan_test_guardcheck")
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_swallowed_violations_still_fail_and_opt_out_marker_works(tmp_path: Path) -> None:
    (tmp_path / "test_probe.py").write_text(
        f"""
import os, socket
from pathlib import Path
import pytest

def test_swallows_network():
    try:
        socket.create_connection(("203.0.113.5", 80), timeout=1)
    except Exception:
        pass

def test_swallows_var_write():
    try:
        Path({str(VAR / PROBE)!r}).write_text("fake")
    except Exception:
        pass

def test_keys_hidden():
    assert "OPENROUTER_API_KEY" not in os.environ

@pytest.mark.allow_paid_keys
def test_keys_opt_in():
    assert os.environ["OPENROUTER_API_KEY"] == "fake-opt-in-value"
""",
        encoding="utf-8",
    )
    result = _pytest(
        [
            "-c",
            str(REPO_ROOT / "pyproject.toml"),
            "--rootdir",
            str(REPO_ROOT),
            "-q",
            str(tmp_path / "test_probe.py"),
        ],
        _clean_env(OPENROUTER_API_KEY="fake-opt-in-value"),
        cwd=tmp_path,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "2 failed, 2 passed" in output
    assert "caught the exception and passed anyway" in output
    assert "203.0.113.5:80" in output and "wrote under the repo's var/" in output
    assert not (VAR / PROBE).exists()
