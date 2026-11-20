"""`python -m portolan.audit` CLI smoke test on the fake path
(`01-spec/build-contracts.md`).

Uses the shared `db` fixture (`tests/conftest.py`) rather than the full
fake-pipeline graph `tests/test_audit.py` builds -- the CLI only needs a
migrated, empty schema to prove it runs end to end without a live model.
"""

from __future__ import annotations

import json

from portolan.audit.__main__ import main


def test_audit_cli_fake_path_runs_on_empty_graph(db, db_url, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("STORE_ROOT", str(tmp_path / "store"))

    exit_code = main(["--since", "2000-01-01", "--rate", "1.0", "--llm", "fake"])

    assert exit_code == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["sources_sampled"] == 0
    assert report["note_slug"].startswith("audit-")


def test_audit_cli_fake_path_runs_with_sampled_sources(db, db_url, monkeypatch, tmp_path, capsys):
    from datetime import UTC, datetime

    from portolan.store import LocalStore

    store_root = tmp_path / "store"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("STORE_ROOT", str(store_root))

    store = LocalStore(store_root)
    store.put("originals/call/S-call-cli-1", b"[00:00:01] Rep: hello there.\n", "text/plain")
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, source_date, ingested_at,
                                  extraction_status)
            VALUES (%s, 'call', %s, %s, %s, 'done')
            """,
            (
                "S-call-cli-1",
                "originals/call/S-call-cli-1",
                "2026-08-01",
                datetime.now(UTC),
            ),
        )

    exit_code = main(["--since", "2000-01-01", "--rate", "1.0", "--llm", "fake"])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sources_sampled"] == 1
    assert report["sources_second_passed"] == 1
    # The fake strong backend finds nothing beyond an empty extraction, and
    # there are no first-pass claims for this hand-inserted source either,
    # so nothing should show up as a miss or invention -- this is a
    # plumbing smoke test, not a correctness test (see tests/test_audit.py
    # for that).
    assert report["genuine_misses_total"] == 0


def test_audit_cli_non_fake_llm_routes_through_get_llm_for_metering(db_url, monkeypatch, tmp_path):
    """Test gap 12 (review finding 11): before this fix, `--llm agent_sdk`
    hand-wired `AgentSDKClient(default_model=..., tenant=...)` directly,
    bypassing `portolan.llm.factory.get_llm`'s real `PostgresUsageSink` --
    the audit's judge call, one of the more expensive recurring jobs,
    never landed a row in `usage_events`. Checked at construction time, no
    network call and no live model needed: the built client's own `.sink`
    must be a `PostgresUsageSink`, not the bare `InMemorySink`
    `AgentSDKClient` defaults to when no sink is given at all.
    """
    from portolan.audit.__main__ import _build_strong_llm
    from portolan.config import get_settings
    from portolan.llm.factory import PostgresUsageSink
    from portolan.metering import InMemorySink

    monkeypatch.setenv("DATABASE_URL", db_url)
    # get_llm() creates store_root; the default is the repo's own var/store (tests/guards.py)
    monkeypatch.setenv("STORE_ROOT", str(tmp_path / "store"))
    settings = get_settings()

    llm = _build_strong_llm("agent_sdk", settings)

    assert isinstance(llm.sink, PostgresUsageSink)
    assert not isinstance(llm.sink, InMemorySink)
    assert llm.sink.database_url == settings.database_url.get_secret_value()
    assert llm.default_model == settings.strong_model
