"""`python -m portolan.consolidate` CLI smoke test on the fake path
(`01-spec/build-contracts.md`).
"""

from __future__ import annotations

from datetime import UTC, datetime

from portolan.consolidate.__main__ import main
from portolan.consolidate.trigger import CONSOLIDATION_CLAIMS_THRESHOLD
from portolan.notes.models import NoteRow, NoteStatus, NoteType
from portolan.notes.persist import read_note, write_note


def _make_note(conn, slug: str) -> NoteRow:
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.ACCOUNT,
        title=slug.title(),
        description="A test account.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        crm_id="A-7000",
    )
    write_note(conn, note, [], changed_by="test", reason="setup")
    return note


def test_consolidate_cli_fake_path_runs_on_empty_graph(db, db_url, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", db_url)

    exit_code = main(["--limit", "5", "--llm", "fake"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.strip() in ("[]", "[\n]")


def test_consolidate_cli_fake_path_consolidates_a_due_note(db, db_url, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", db_url)

    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO sources (slug, source_type, s3_key, source_date, extraction_status) "
            "VALUES ('s1', 'call', 'originals/call/s1', '2026-08-01', 'done')"
        )
    _make_note(db, "cli-note")
    with db.cursor() as cur:
        for i in range(CONSOLIDATION_CLAIMS_THRESHOLD + 1):
            cur.execute(
                "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator) "
                "VALUES (%s, 'complaint', %s, 's1', %s)",
                ("cli-note", f"Fact {i}.", f"{i:02d}:00:00"),
            )

    exit_code = main(["--limit", "5", "--llm", "fake"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert '"status":"consolidated"' in out.replace(" ", "") or "consolidated" in out

    note, claims = read_note(db, "cli-note")
    assert len(claims) == CONSOLIDATION_CLAIMS_THRESHOLD
    assert note.synthesis


def test_consolidate_cli_non_fake_llm_routes_through_get_llm_for_metering(
    db_url, monkeypatch, tmp_path
):
    """Test gap 12 (review finding 11): before this fix, `--llm agent_sdk`
    hand-wired an `AgentSDKClient` pair directly, bypassing `portolan.llm.
    factory.get_llm`'s real `PostgresUsageSink` -- consolidation's
    strong-model retries, one of the more expensive recurring jobs, never
    landed a row in `usage_events`. Checked at construction time, no
    network call and no live model needed: both built clients' `.sink`
    must be a `PostgresUsageSink`, and each must carry its own model
    (`compare_model` for the cheap pass, `strong_model` for the retry),
    not the bare `InMemorySink`/single hard-coded model the old hand-wired
    pair used.
    """
    from portolan.config import get_settings
    from portolan.consolidate.__main__ import _build_llms
    from portolan.llm.factory import PostgresUsageSink
    from portolan.metering import InMemorySink

    monkeypatch.setenv("DATABASE_URL", db_url)
    # get_llm() creates store_root; the default is the repo's own var/store (tests/guards.py)
    monkeypatch.setenv("STORE_ROOT", str(tmp_path / "store"))
    settings = get_settings()

    cheap, strong = _build_llms("agent_sdk", settings)

    for llm in (cheap, strong):
        assert isinstance(llm.sink, PostgresUsageSink)
        assert not isinstance(llm.sink, InMemorySink)
        assert llm.sink.database_url == settings.database_url.get_secret_value()

    assert cheap.default_model == settings.compare_model
    assert strong.default_model == settings.strong_model
