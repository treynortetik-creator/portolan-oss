"""Tests for the `db` fixture's own safety guard (review finding F18) and
for `with_database_name`, the DSN-rewriting helper CI's postgres:5432
found a hole in (2026-09-14 CI red).

`_database_name` is the pure-function half of the guard (extracting a
database name out of a DSN, no I/O); the fixture's actual refusal against
a real non-test database is exercised by hand, not here, since triggering
it means pointing DATABASE_URL at a database this suite would then refuse
to touch -- exactly the scenario the guard exists to make impossible to
do by accident. Testing the string-extraction logic it depends on is the
useful, safe part to cover with an automated test.

`with_database_name` used to be three copy-pasted bare URLs
(`postgresql://localhost/portolan_test_pipeline`) in
`tests/test_pipeline.py`, which silently dropped `DATABASE_URL`'s
user/password/host/port -- invisible on a trust-auth dev Postgres, fatal
in CI (`fe_sendauth: no password supplied`). These tests are the
"a DSN with credentials survives the rewrite" regression test for that.
"""

from __future__ import annotations

from conftest import _database_name, with_database_name


def test_database_name_extracts_the_simple_case() -> None:
    assert _database_name("postgresql://localhost/portolan_test_fix") == "portolan_test_fix"


def test_database_name_ignores_credentials_host_and_port() -> None:
    assert (
        _database_name("postgresql://user:pw@db.example.test:5432/portolan_test_review")
        == "portolan_test_review"
    )


def test_database_name_strips_query_string() -> None:
    assert (
        _database_name("postgresql://localhost/portolan_test_fix?sslmode=require")
        == "portolan_test_fix"
    )


def test_database_name_of_a_non_test_database_does_not_contain_test() -> None:
    # This is exactly the shape the F18 guard refuses to run against.
    assert "test" not in _database_name("postgresql://localhost/portolan_production").lower()


def test_with_database_name_preserves_user_password_host_and_port() -> None:
    # The CI regression: a bare "postgresql://localhost/<name>" rewrite
    # dropped `portolan:portolan@...:5432`, and psycopg then tried to
    # authenticate over TCP with no password at all.
    rewritten = with_database_name(
        "postgresql://portolan:portolan@localhost:5432/portolan_test", "portolan_test_pipeline"
    )
    assert rewritten == "postgresql://portolan:portolan@localhost:5432/portolan_test_pipeline"


def test_with_database_name_preserves_query_string_and_fragment() -> None:
    rewritten = with_database_name(
        "postgresql://user:pw@db.example.test:5432/portolan_test?sslmode=require#frag",
        "portolan_test_other",
    )
    assert rewritten == (
        "postgresql://user:pw@db.example.test:5432/portolan_test_other?sslmode=require#frag"
    )


def test_with_database_name_on_a_bare_local_url_needs_no_credentials() -> None:
    # Local trust-auth Postgres never had a password to lose in the first
    # place -- this is the shape that let the bug hide on a dev machine.
    assert (
        with_database_name("postgresql://localhost/portolan_test", "portolan_test_pipeline")
        == "postgresql://localhost/portolan_test_pipeline"
    )
