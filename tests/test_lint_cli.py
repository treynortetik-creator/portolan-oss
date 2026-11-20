"""Tests for the `python -m portolan.lint` CLI (spec sec 6.1).

Runs `portolan.lint.__main__.main()` in-process against the `db` fixture's
database (both this test and `main()` read `DATABASE_URL` from the same
environment, and the fixture's rows are committed immediately since its
connection is autocommit — see `tests/conftest.py`), rather than shelling
out, so it stays fast and gets coverage.

The store root is set via `monkeypatch.setenv("STORE_ROOT", ...)` (polish
wave 2026-09-12): the CLI used to read a `PORTOLAN_STORE_ROOT` variable
nobody else in the codebase reads or sets, so it silently ignored whatever
store root `Settings`/the rest of the app were actually configured with.
`test_cli_against_a_demo_graph_with_real_originals_reports_zero_missing_
original` is this fix's own proof: real files on disk under `STORE_ROOT`,
`sources` rows pointing at them, and a `0` count in the `missing_original`
row of the rendered report.
"""

from __future__ import annotations

from portolan.lint.__main__ import main
from portolan.store import LocalStore
from portolan.util import clock


def test_cli_exits_zero_and_writes_report_on_a_clean_graph(db, tmp_path, monkeypatch):
    monkeypatch.setenv("STORE_ROOT", str(tmp_path / "store"))

    exit_code = main(["--out", str(tmp_path / "lint")])

    assert exit_code == 0
    out_file = tmp_path / "lint" / f"lint-{clock.today():%Y-%m-%d}.md"
    assert out_file.is_file()
    assert "0 error(s)" in out_file.read_text()

    with db.cursor() as cur:
        cur.execute("SELECT kind, summary FROM changelog ORDER BY id DESC LIMIT 1")
        kind, summary = cur.fetchone()
    assert kind == "lint"
    assert "0 error(s)" in summary


def test_cli_exits_one_when_the_graph_has_an_error_finding(db, tmp_path, monkeypatch):
    monkeypatch.setenv("STORE_ROOT", str(tmp_path / "store"))

    with db.cursor() as cur:
        # A living account note with no inbound links: an `orphan` error.
        cur.execute(
            """
            INSERT INTO notes (slug, type, title, description, crm_id)
            VALUES ('ashgrove', 'account', 'Ashgrove Health',
                    'A hospital system account.', 'A-1010')
            """
        )

    exit_code = main(["--out", str(tmp_path / "lint")])

    assert exit_code == 1
    out_file = tmp_path / "lint" / f"lint-{clock.today():%Y-%m-%d}.md"
    contents = out_file.read_text()
    assert "error(s)" in contents
    assert "ashgrove" in contents


def test_cli_writes_the_report_under_the_deployment_clocks_date(db, tmp_path, monkeypatch):
    """P2-9 (bug hunt 2026-09-18): the CLI's own `today = clock.today()`
    (not the stdlib `date.today()`) decides the output filename. Proven
    by monkeypatching `clock.today` to a deliberately-unreal date and
    checking the CLI wrote its report there -- before the fix, `__main__.
    py` called the stdlib `date.today()` directly and this monkeypatch
    would have had no effect on the filename at all."""
    from datetime import date

    monkeypatch.setenv("STORE_ROOT", str(tmp_path / "store"))
    sentinel = date(2099, 1, 2)
    monkeypatch.setattr(clock, "today", lambda: sentinel)

    exit_code = main(["--out", str(tmp_path / "lint")])

    assert exit_code == 0
    out_file = tmp_path / "lint" / f"lint-{sentinel:%Y-%m-%d}.md"
    assert out_file.is_file()
    assert out_file.read_text().startswith(f"# Lint report — {sentinel:%Y-%m-%d}")


def test_cli_against_a_demo_graph_with_real_originals_reports_zero_missing_original(
    db, tmp_path, monkeypatch
):
    """Proves the fix, not just the env var name: a `sources` row whose
    `s3_key` genuinely exists under `STORE_ROOT` must not be reported as
    `missing_original` -- the failure mode this fix closes was the CLI
    resolving a *different* store root than the one its files actually
    live under and reporting every one of them missing."""
    store_root = tmp_path / "store"
    monkeypatch.setenv("STORE_ROOT", str(store_root))

    store = LocalStore(store_root)
    store.put("originals/call/S-call-900", b"hello world", "text/plain")

    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, content_hash,
                                  source_date, extraction_status)
            VALUES ('S-call-900', 'call', 'originals/call/S-call-900',
                    'fake-hash-900', '2026-08-01', 'done')
            """
        )

    exit_code = main(["--out", str(tmp_path / "lint")])

    assert exit_code == 0
    out_file = tmp_path / "lint" / f"lint-{clock.today():%Y-%m-%d}.md"
    contents = out_file.read_text()
    assert "| missing_original | 0 | 0 |" in contents
    assert "0 error(s)" in contents
