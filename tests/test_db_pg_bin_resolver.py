"""Unit tests for `portolan.db._resolve_pg_binary` and its helpers, the
fix for the `pg_dump`/`pg_restore` server-version-mismatch bug: on this
project's own dev Mac, Homebrew's `pg_dump` on PATH is 16.14 while
`brew services` runs a 17.11 server -- `pg_dump` refuses to dump a server
newer than itself ("server version mismatch"), and the same gap exists on
a GitHub Actions runner against the `pgvector/pgvector:pg17` CI service.

These tests use fake shell-script "binaries" (a `--version` line is all
`_binary_version` reads) and a monkeypatched `PATH`, never a real
Postgres connection -- `_resolve_pg_binary` takes `server_major` as a
plain int and an explicit `search_dirs` list, so nothing here needs
`tests/conftest.py`'s `db`/`db_url` fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portolan import db as db_module


def _make_fake_binary(path: Path, version_line: str) -> None:
    """Write an executable shell script at `path` that prints
    `version_line` no matter what arguments it's called with -- enough to
    stand in for `pg_dump --version` / `pg_restore --version`, which is
    all `_binary_version` reads."""
    path.write_text(f"#!/bin/sh\necho '{version_line}'\n")
    path.chmod(0o755)


def test_prefers_a_matching_homebrew_style_directory_over_an_older_path_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PATH has only the older, non-matching pg_dump (this machine's own
    # real-world bug: Homebrew PATH pg_dump 16.14 against a 17.11 server).
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    _make_fake_binary(path_bin / "pg_dump", "pg_dump (PostgreSQL) 16.14 (Homebrew)")
    monkeypatch.setenv("PATH", str(path_bin))

    # A matching binary sits in a "Homebrew-style" versioned directory.
    homebrew_dir = tmp_path / "postgresql@17" / "bin"
    homebrew_dir.mkdir(parents=True)
    _make_fake_binary(homebrew_dir / "pg_dump", "pg_dump (PostgreSQL) 17.11 (Homebrew)")

    resolved = db_module._resolve_pg_binary("pg_dump", 17, search_dirs=[homebrew_dir])

    assert resolved == str(homebrew_dir / "pg_dump")


def test_skips_a_non_matching_search_dir_and_falls_through_to_a_later_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    (tmp_path / "empty-bin").mkdir()

    wrong_major_dir = tmp_path / "postgresql@16" / "bin"
    wrong_major_dir.mkdir(parents=True)
    _make_fake_binary(wrong_major_dir / "pg_dump", "pg_dump (PostgreSQL) 16.14")

    right_major_dir = tmp_path / "postgresql@17" / "bin"
    right_major_dir.mkdir(parents=True)
    _make_fake_binary(right_major_dir / "pg_dump", "pg_dump (PostgreSQL) 17.4")

    resolved = db_module._resolve_pg_binary(
        "pg_dump", 17, search_dirs=[wrong_major_dir, right_major_dir]
    )

    assert resolved == str(right_major_dir / "pg_dump")


def test_falls_back_to_a_matching_path_binary_when_no_search_dir_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    _make_fake_binary(path_bin / "pg_dump", "pg_dump (PostgreSQL) 17.4")
    monkeypatch.setenv("PATH", str(path_bin))

    resolved = db_module._resolve_pg_binary("pg_dump", 17, search_dirs=[])

    assert resolved == str(path_bin / "pg_dump")


def test_raises_a_clear_error_naming_both_versions_when_nothing_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    _make_fake_binary(path_bin / "pg_dump", "pg_dump (PostgreSQL) 16.14 (Homebrew)")
    monkeypatch.setenv("PATH", str(path_bin))

    with pytest.raises(db_module.BackupError) as excinfo:
        db_module._resolve_pg_binary("pg_dump", 17, search_dirs=[])

    message = str(excinfo.value)
    assert "17" in message  # the server's major version
    assert "16.14" in message  # the mismatched PATH version
    assert "PG_BIN_DIR" in message  # the setting to fix it


def test_raises_a_clear_error_when_path_has_nothing_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    (tmp_path / "empty-bin").mkdir()

    with pytest.raises(db_module.BackupError, match="pg_dump"):
        db_module._resolve_pg_binary("pg_dump", 17, search_dirs=[])


def test_pg_bin_search_dirs_puts_the_configured_pg_bin_dir_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PG_BIN_DIR", str(tmp_path / "custom-bin"))
    dirs = db_module._pg_bin_search_dirs(17)
    assert dirs[0] == tmp_path / "custom-bin"
    # The well-known locations still follow, formatted with the major
    # version passed in.
    assert Path("/opt/homebrew/opt/postgresql@17/bin") in dirs


def test_pg_bin_search_dirs_omits_pg_bin_dir_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PG_BIN_DIR", raising=False)
    dirs = db_module._pg_bin_search_dirs(17)
    assert dirs == [
        Path("/opt/homebrew/opt/postgresql@17/bin"),
        Path("/usr/lib/postgresql/17/bin"),
        Path("/usr/local/opt/postgresql@17/bin"),
    ]
