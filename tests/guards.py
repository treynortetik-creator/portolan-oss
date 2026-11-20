"""Test-suite guardrails, loaded as a pytest plugin (guardrails plan 2026-09-22, ranks 1 and 3).

Loaded twice over, on purpose: `pyproject.toml` passes `-p tests.guards` (so `--noconftest`
does not skip it) and `tests/conftest.py` registers it when the ini file was bypassed
(`-c /dev/null`). `-p no:tests.guards` is refused outright by conftest.

1. Test database. The run refuses to start (`pytest.exit`, code 4) when `DATABASE_URL` -- or
   the libpq variables that can redirect it (`PGHOST`, `PGHOSTADDR`, `PGDATABASE`,
   `PGSERVICE`) -- point anywhere but a local database whose name starts with
   `portolan_test` (the bare name itself only when `GITHUB_ACTIONS=true`: it was the dev
   graph's name), or when any `RAILWAY_*` variable is set (what `railway run` injects).
   Every Postgres connection the process opens afterwards is checked the same way at
   `psycopg.generators.connect`, the one function every psycopg connect path goes through,
   so code under test that falls back to `.env` or a default DSN cannot reach a dev or live
   database either. The only other name allowed per connection is the `postgres`
   maintenance database, which the fixtures use to CREATE/DROP their own test databases.
   Local means localhost, a loopback IP, or a unix socket; inside a container (`/.dockerenv`)
   also `db`, the service name in `docker/docker-compose.yml`. CI's `test` job uses
   localhost (`.github/workflows/ci.yml`). Messages never print a DSN or a host.

2. Network. `socket.socket.connect` / `connect_ex` refuse anything but loopback and unix
   sockets, with a message naming the test (2026-09-22: a test called the live OpenRouter
   key endpoint). libpq opens its own sockets in C, so Postgres is untouched by this.

3. Paid-model keys. `OPENROUTER_API_KEY`, `TYPESAFE_API_KEY` and `ANTHROPIC_API_KEY` are
   removed from the environment for the whole run, and their lines are filtered out of any
   read of the repo's `.env` (pydantic-settings and `_env_or_dotenv` both read it directly),
   unless a test carries `@pytest.mark.allow_paid_keys`. The network guard still applies.

4. Repo `var/`. Writes under the repo's `var/` (and the main checkout's, from a worktree)
   raise: `open`/`io.open` in a write mode, `os.open` with a write flag, `os.mkdir`
   (so `os.makedirs` and `Path.mkdir`), rename/replace/remove/unlink/rmdir/link/symlink/
   truncate, and `shutil.rmtree` (a `logging.FileHandler` opens through the patched `open`).
   Intercepted in-process because live data jobs write under `var/data/` while tests run,
   so a before/after snapshot would false-positive (2026-09-21: tests wrote fake records
   into `var/data`).

A guard hit raises `GuardViolation` at the call. It is also recorded, and a test that
swallowed the exception (`except Exception: pass`) is still failed by
`pytest_runtest_makereport`.

Limits: only this process is covered. A subprocess the test starts is not.
"""

from __future__ import annotations

import builtins
import contextlib
import getpass
import io
import ipaddress
import os
import re
import shutil
import socket
from pathlib import Path

import psycopg.generators as _pg_generators
import pytest
from psycopg.conninfo import conninfo_to_dict

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DB_PREFIX = "portolan_test"
# The bare prefix is the name `.env.example` gave the dev graph until 2026-09-22, so a dev
# database may still carry it: refused unless CI_MARKER says this is a CI runner (GitHub sets
# it on every Actions job; CI's `test` job uses exactly this name).
BARE_TEST_DB = TEST_DB_PREFIX
CI_MARKER = ("GITHUB_ACTIONS", "true")
MAINTENANCE_DBS = frozenset({"postgres"})
CONTAINER_DB_HOSTS = frozenset({"db"})  # docker/docker-compose.yml service name
# What the session check assumes when DATABASE_URL is unset: conftest's per-process name.
DEFAULT_TEST_DATABASE_URL = f"postgresql://localhost/portolan_test_{os.getpid()}"
PAID_MODEL_KEYS = ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "ANTHROPIC_API_KEY")
RUN_IT = (
    "env -u OPENROUTER_API_KEY -u DATABASE_URL TZ=UTC PGTZ=UTC PORTOLAN_ROLE=test uv run pytest -q"
)
_LIBPQ_ENV = (
    ("host", "PGHOST"),
    ("hostaddr", "PGHOSTADDR"),
    ("dbname", "PGDATABASE"),
    ("service", "PGSERVICE"),
    ("user", "PGUSER"),
)


class GuardViolation(RuntimeError):
    """A test tried to do something the suite's guardrails forbid."""


def main_checkout_root(repo_root: Path) -> Path | None:
    """`<main>/.claude/worktrees/<name>` -> `<main>`; None outside a worktree."""
    parts = repo_root.parts
    for i in range(len(parts) - 2):
        if parts[i] == ".claude" and parts[i + 1] == "worktrees":
            return Path(*parts[:i])
    return None


# --------------------------------------------------------------------------- state


class _State:
    def __init__(self) -> None:
        self.installed = False
        self.current_test: str | None = None
        self.violations: list[str] = []
        self.stashed_keys: dict[str, str] = {}
        self.hide_dotenv_keys = True
        roots = [REPO_ROOT]
        main = main_checkout_root(REPO_ROOT)
        if main is not None:
            roots.append(main)
        self.var_roots = tuple(os.path.realpath(r / "var") for r in roots)
        self.dotenv_paths = {os.path.realpath(r / ".env") for r in roots}
        self.saved: list[tuple[object, str, object]] = []


STATE = _State()


def _who() -> str:
    return (
        f"test {STATE.current_test}"
        if STATE.current_test
        else "code outside any test (collection or a fixture)"
    )


def _violate(message: str) -> GuardViolation:
    full = f"{_who()} {message}"
    STATE.violations.append(full)
    return GuardViolation(full)


def take_violations() -> list[str]:
    found, STATE.violations = STATE.violations, []
    return found


# --------------------------------------------------------------------------- 1. database


def in_container() -> bool:
    return os.path.exists("/.dockerenv")


def host_is_local(host: str, *, container: bool) -> bool:
    h = host.strip()
    if h == "" or h.startswith("/") or h.startswith("@"):
        return True  # unix socket (default dir, a directory, or a Linux abstract socket)
    h = h.strip("[]").lower().rstrip(".")
    if h == "localhost":
        return True
    if container and h in CONTAINER_DB_HOSTS:
        return True
    try:
        ip = ipaddress.ip_address(h.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or bool(mapped and mapped.is_loopback)


def effective_params(conninfo: str, environ) -> dict[str, str]:
    """What libpq will actually use: the DSN's own keys, then the PG* variables for the
    ones it leaves out. Raises on a DSN that does not parse."""
    params = {k: str(v) for k, v in conninfo_to_dict(conninfo or "").items() if v is not None}
    for key, var in _LIBPQ_ENV:
        if not params.get(key) and environ.get(var):
            params[key] = environ[var]
    return params


def database_refusal(
    conninfo: str, environ, *, container: bool, allow_maintenance: bool
) -> str | None:
    """None if a connection with this DSN (plus environ) stays on a local test database,
    else the reason, worded so it never includes the DSN, a host or a password."""
    try:
        params = effective_params(conninfo, environ)
    except Exception:
        return "does not parse as a Postgres connection string"
    if params.get("service"):
        return "uses a libpq service (PGSERVICE / service=), which can point anywhere"
    dbname = params.get("dbname") or params.get("user") or getpass.getuser()
    if not dbname.startswith(TEST_DB_PREFIX) and not (
        allow_maintenance and dbname in MAINTENANCE_DBS
    ):
        return (
            f"names database {dbname!r}; tests only run against a database whose name "
            f"starts with {TEST_DB_PREFIX!r}"
        )
    hosts = (params.get("host") or "").split(",")
    for host in hosts:
        if not host_is_local(host, container=container):
            return "names a host that is not local (localhost, a loopback IP or a unix socket)"
    addrs = (params.get("hostaddr") or "").split(",")
    for i, addr in enumerate(addrs):
        addr = addr.strip()
        if not addr or host_is_local(addr, container=False):
            continue
        # psycopg resolves `host` into `hostaddr` before libpq connects; inside a container
        # the compose service `db` resolves to its private bridge address.
        host = (hosts[i] if i < len(hosts) else hosts[-1]).strip().lower()
        if container and host in CONTAINER_DB_HOSTS and _is_private_ip(addr):
            continue
        return "names a hostaddr that is not a loopback IP"
    if dbname == BARE_TEST_DB and environ.get(CI_MARKER[0]) != CI_MARKER[1]:
        return (
            f"names database {dbname!r}, the old dev-graph name, outside CI; locally tests "
            f"use {TEST_DB_PREFIX}_<something> (unset DATABASE_URL for the per-process one)"
        )
    return None


def _is_private_ip(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr.split("%", 1)[0]).is_private
    except ValueError:
        return False


def session_refusal(environ, *, container: bool) -> str | None:
    railway = sorted(k for k in environ if k.startswith("RAILWAY_"))
    if railway:
        shown = ", ".join(railway[:4]) + (", ..." if len(railway) > 4 else "")
        return (
            f"{len(railway)} RAILWAY_* variable(s) are set ({shown}). That is what `railway run` "
            "injects, so this shell can reach the live deployment."
        )
    url = environ.get("DATABASE_URL")
    reason = database_refusal(
        url or DEFAULT_TEST_DATABASE_URL, environ, container=container, allow_maintenance=False
    )
    if reason is None:
        return None
    return (
        f"DATABASE_URL {reason}."
        if url
        else f"The default test database {reason} (check the PG* variables)."
    )


def refusal_message(reason: str) -> str:
    return (
        f"Refusing to run the test suite: {reason}\n"
        f"Run it with the command in CLAUDE.md 'Run it': {RUN_IT} <paths>\n"
        "(it unsets DATABASE_URL, and the suite then creates its own local portolan_test_<pid> "
        "database). Guard: tests/guards.py."
    )


_real_pg_connect = _pg_generators.connect


def _guarded_pg_connect(conninfo, *args, **kwargs):
    reason = database_refusal(
        conninfo, os.environ, container=in_container(), allow_maintenance=True
    )
    if reason:
        raise _violate(
            f"tried to open a Postgres connection that {reason}. Use the `db` / `db_url` "
            "fixtures (tests/conftest.py), never a DSN from .env or a default."
        )
    return _real_pg_connect(conninfo, *args, **kwargs)


# --------------------------------------------------------------------------- 2. network

_real_sock_connect = socket.socket.connect
_real_sock_connect_ex = socket.socket.connect_ex


def address_is_local(family: int, address) -> bool:
    if family not in (socket.AF_INET, socket.AF_INET6):
        return True  # AF_UNIX and friends never leave the machine
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip("[]").lower().rstrip(".")
    if host in ("", "localhost"):
        return True  # "" is INADDR_ANY, which connect() treats as this machine
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False  # any other hostname
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or ip.is_unspecified or bool(mapped and mapped.is_loopback)


def _check_socket(sock: socket.socket, address) -> None:
    if address_is_local(sock.family, address):
        return
    host, port = (address[0], address[1]) if isinstance(address, tuple) else (address, "?")
    raise _violate(
        f"tried to open a network connection to {host}:{port}. Tests may only reach loopback "
        "and unix sockets: stub the call (an httpx.MockTransport, a fake client) instead."
    )


def _guarded_connect(self, address):
    _check_socket(self, address)
    return _real_sock_connect(self, address)


def _guarded_connect_ex(self, address):
    _check_socket(self, address)
    return _real_sock_connect_ex(self, address)


# --------------------------------------------------------------------------- 3 + 4. files

_real_open = builtins.open
_real_os_open = os.open
_real_mkdir = os.mkdir
_real_rename = os.rename
_real_replace = os.replace
_real_remove = os.remove
_real_unlink = os.unlink
_real_rmdir = os.rmdir
_real_truncate = os.truncate
_real_symlink = os.symlink
_real_link = os.link
_real_rmtree = shutil.rmtree
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_PAID_KEY_LINE = re.compile(
    rb"^[ \t]*(?:export[ \t]+)?(?:"
    + "|".join(PAID_MODEL_KEYS).encode()
    + rb")[ \t]*=.*(?:\r?\n|$)",
    re.M,
)


def _resolve(path) -> str | None:
    if isinstance(path, int):
        return None
    try:
        raw = os.fsdecode(os.fspath(path))
    except TypeError:
        return None
    return os.path.realpath(os.path.abspath(raw))


def under_var(path) -> str | None:
    resolved = _resolve(path)
    if resolved is None:
        return None
    for root in STATE.var_roots:
        if resolved == root or resolved.startswith(root + os.sep):
            return resolved
    return None


def _check_write(path, how: str) -> None:
    resolved = under_var(path)
    if resolved is None:
        return
    shown = resolved
    for base in (REPO_ROOT, main_checkout_root(REPO_ROOT)):
        if base is not None and resolved.startswith(str(base) + os.sep):
            shown = os.path.relpath(resolved, base)
            break
    raise _violate(
        f"wrote under the repo's var/ ({shown}) via {how}. var/ holds real data that live jobs "
        "write while tests run (2026-09-21: tests wrote fake records into var/data); point "
        "the code at tmp_path instead."
    )


def _is_write_mode(mode) -> bool:
    return isinstance(mode, str) and any(ch in mode for ch in "wax+")


def _is_guarded_dotenv(file) -> bool:
    if isinstance(file, int):
        return False
    try:
        raw = os.fsdecode(os.fspath(file))
    except TypeError:
        return False
    if os.path.basename(raw) != ".env":
        return False
    return os.path.realpath(os.path.abspath(raw)) in STATE.dotenv_paths


def _filtered_dotenv(file, mode, args, kwargs):
    with _real_open(file, "rb") as fh:
        data = _PAID_KEY_LINE.sub(b"", fh.read())
    if "b" in mode:
        return io.BytesIO(data)
    encoding = kwargs.get("encoding") or (args[1] if len(args) > 1 else None) or "utf-8"
    errors = kwargs.get("errors") or (args[2] if len(args) > 2 else None) or "strict"
    return io.StringIO(data.decode(encoding, errors))


def _guarded_open(file, mode="r", *args, **kwargs):
    if _is_write_mode(mode):
        _check_write(file, f"open(mode={mode!r})")
    elif STATE.hide_dotenv_keys and _is_guarded_dotenv(file):
        return _filtered_dotenv(file, mode, args, kwargs)
    return _real_open(file, mode, *args, **kwargs)


def _guarded_os_open(path, flags, mode=0o777, *, dir_fd=None):
    if dir_fd is None and flags & _WRITE_FLAGS:
        _check_write(path, "os.open")
    return _real_os_open(path, flags, mode, dir_fd=dir_fd)


def _guarded_mkdir(path, mode=0o777, *, dir_fd=None):
    if dir_fd is None:
        _check_write(path, "os.mkdir / os.makedirs / Path.mkdir")
    return _real_mkdir(path, mode, dir_fd=dir_fd)


def _two_path(real, how):
    def wrapper(src, dst, *args, **kwargs):
        if kwargs.get("src_dir_fd") is None:
            _check_write(src, how)
        if kwargs.get("dst_dir_fd") is None:
            _check_write(dst, how)
        return real(src, dst, *args, **kwargs)

    return wrapper


def _dst_only(real, how):
    def wrapper(src, dst, *args, **kwargs):
        if kwargs.get("dir_fd") is None and kwargs.get("dst_dir_fd") is None:
            _check_write(dst, how)
        return real(src, dst, *args, **kwargs)

    return wrapper


def _one_path(real, how):
    def wrapper(path, *args, **kwargs):
        if kwargs.get("dir_fd") is None:
            _check_write(path, how)
        return real(path, *args, **kwargs)

    return wrapper


def _guarded_rmtree(path, *args, **kwargs):
    if kwargs.get("dir_fd") is None:
        _check_write(path, "shutil.rmtree")
    return _real_rmtree(path, *args, **kwargs)


# --------------------------------------------------------------------------- install


def _patch(target, name: str, value) -> None:
    STATE.saved.append((target, name, getattr(target, name)))
    setattr(target, name, value)


def install() -> None:
    if STATE.installed:
        return
    STATE.installed = True
    for name in PAID_MODEL_KEYS:
        if name in os.environ:
            STATE.stashed_keys[name] = os.environ.pop(name)
    _patch(_pg_generators, "connect", _guarded_pg_connect)
    _patch(socket.socket, "connect", _guarded_connect)
    _patch(socket.socket, "connect_ex", _guarded_connect_ex)
    _patch(builtins, "open", _guarded_open)
    _patch(io, "open", _guarded_open)
    _patch(os, "open", _guarded_os_open)
    _patch(os, "mkdir", _guarded_mkdir)
    _patch(os, "rename", _two_path(_real_rename, "os.rename"))
    _patch(os, "replace", _two_path(_real_replace, "os.replace"))
    _patch(os, "remove", _one_path(_real_remove, "os.remove"))
    _patch(os, "unlink", _one_path(_real_unlink, "os.unlink"))
    _patch(os, "rmdir", _one_path(_real_rmdir, "os.rmdir"))
    _patch(os, "truncate", _one_path(_real_truncate, "os.truncate"))
    _patch(os, "symlink", _dst_only(_real_symlink, "os.symlink"))
    _patch(os, "link", _dst_only(_real_link, "os.link"))
    _patch(shutil, "rmtree", _guarded_rmtree)


def uninstall() -> None:
    while STATE.saved:
        target, name, value = STATE.saved.pop()
        setattr(target, name, value)
    for name, value in STATE.stashed_keys.items():
        os.environ.setdefault(name, value)
    STATE.stashed_keys.clear()
    STATE.installed = False


@contextlib.contextmanager
def dotenv_keys_visible():
    previous, STATE.hide_dotenv_keys = STATE.hide_dotenv_keys, False
    try:
        yield
    finally:
        STATE.hide_dotenv_keys = previous


# --------------------------------------------------------------------------- pytest hooks


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "allow_paid_keys: keep OPENROUTER_API_KEY, TYPESAFE_API_KEY and ANTHROPIC_API_KEY "
        "(environment and .env) visible to this test. Network is still blocked.",
    )
    reason = session_refusal(os.environ, container=in_container())
    if reason:
        pytest.exit(refusal_message(reason), returncode=4)
    install()


def pytest_unconfigure(config: pytest.Config) -> None:
    uninstall()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    STATE.current_test = item.nodeid
    take_violations()
    try:
        return (yield)
    finally:
        STATE.current_test = None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    swallowed = take_violations()
    if swallowed and report.passed:
        report.outcome = "failed"
        report.longrepr = (
            "test hygiene guard (tests/guards.py) fired, and the test caught the exception "
            "and passed anyway:\n" + "\n".join(swallowed)
        )
    return report


@pytest.fixture(autouse=True)
def _paid_model_keys(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if request.node.get_closest_marker("allow_paid_keys"):
        for name, value in STATE.stashed_keys.items():
            monkeypatch.setenv(name, value)
        with dotenv_keys_visible():
            yield
        return
    for name in PAID_MODEL_KEYS:
        monkeypatch.delenv(name, raising=False)
    yield
