"""`portolan.report.executive` tests (weekly-report wave 2, "make the
weekly report a real executive summary"): `build_executive_summary`'s
own mechanical gates (citation validation, the numbers rule, the
account-slug allowlist, the line cap) and its `failure_reasons` contract
-- the same "drop it, don't patch it" discipline `weekly._headline`
already has, applied to the new section.

Uses the lightweight, per-test `db` fixture (`tests/conftest.py`,
per-process isolated) rather than `test_report_helpers.py`'s expensive,
session-scoped `full_graph` -- these tests only need a handful of hand-
planted claims, not the whole synthetic corpus.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from portolan.llm.fake import FakeLLMClient
from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
from portolan.report.executive import EXECUTIVE_SYSTEM, MAX_RENDERED_LINES, build_executive_summary
from portolan.report.models import (
    ActionRow,
    ActionsFeed,
    CoverageSummary,
    DiffFeed,
    QuietAccounts,
    ThemeReport,
    WavesFeed,
)

SINCE = date(2026, 8, 17)
AS_OF = date(2026, 8, 24)


def _bare_feeds():
    diff = DiffFeed(since=SINCE, as_of=AS_OF, quiet_accounts=QuietAccounts(as_of=AS_OF))
    waves = WavesFeed(since=SINCE, as_of=AS_OF)
    actions = ActionsFeed()
    coverage = CoverageSummary(since=SINCE, as_of=AS_OF)
    themes = ThemeReport(since=SINCE, as_of=AS_OF)
    return diff, waves, actions, coverage, themes


def _seed_account_claim(
    conn,
    *,
    account_slug: str = "acme-health",
    crm_id: str = "CRM-ACME",
    source_slug: str = "S-exec-001",
    locator: str = "00:01:00",
    claim_text: str = "The account raised a concern about onboarding delays.",
    fact_type: str = "complaint",
    recorded: datetime | None = None,
) -> tuple[str, str]:
    """A minimal account note plus one claim living directly on it
    (`home_account_slug`'s case 1 -- the home note IS the account note),
    so `build_executive_summary`'s own `_window_claims`/`_account_context`
    have something real to work with. Returns `(citation_marker,
    account_slug)`."""
    recorded = recorded or datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sources (slug, source_type, source_date, extraction_status, s3_key) "
            "VALUES (%s, 'call', %s, 'done', %s) ON CONFLICT (slug) DO NOTHING",
            (source_slug, recorded.date(), f"originals/{source_slug}"),
        )
    now = datetime.now(UTC)
    note = NoteRow(
        slug=account_slug,
        type=NoteType.ACCOUNT,
        title="Acme Health",
        description="Test fixture account.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        crm_id=crm_id,
    )
    write_note(conn, note, [], changed_by="test", reason="executive-summary fixture account")
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (account_slug, fact_type, claim_text, source_slug, locator, recorded),
        )
    conn.commit()
    return f"^[{source_slug} @ {locator}]", account_slug


def _call(conn, llm, **overrides) -> tuple:
    diff, waves, actions, coverage, themes = _bare_feeds()
    failure_reasons: list[str] = []
    result = build_executive_summary(
        conn,
        llm,
        since=SINCE,
        as_of=AS_OF,
        diff=diff,
        themes=themes,
        waves=waves,
        questions=[],
        outliers=[],
        contact_changes=[],
        actions=actions,
        coverage=coverage,
        failure_reasons=failure_reasons,
        **overrides,
    )
    return result, failure_reasons


def test_happy_path_keeps_a_cited_bullet_and_a_known_account_paragraph(db) -> None:
    citation, account_slug = _seed_account_claim(db)

    def _responder(*, system, user, json_schema, model, surface):
        del user, json_schema, model, surface
        assert system == EXECUTIVE_SYSTEM
        return {
            "this_week": ["Onboarding delays surfaced at one account this week."],
            "attention": [f"Acme Health flagged onboarding delays. {citation}"],
            "by_account": [
                {"account_slug": account_slug, "paragraph": "Acme Health needs a follow-up call."}
            ],
            "suggested_actions": ["Schedule a call with Acme Health about onboarding."],
        }

    result, _reasons = _call(db, FakeLLMClient(_responder))
    assert result.failure_reason is None
    assert result.this_week == ["Onboarding delays surfaced at one account this week."]
    assert result.attention == [f"Acme Health flagged onboarding delays. {citation}"]
    assert len(result.by_account) == 1
    assert result.by_account[0].account_slug == account_slug
    assert result.suggested_actions == ["Schedule a call with Acme Health about onboarding."]


def test_drops_a_bullet_with_an_unsupported_citation(db) -> None:
    _citation, _account_slug = _seed_account_claim(db)

    def _responder(*, system, user, json_schema, model, surface):
        del user, json_schema, model, surface
        return {
            "this_week": ["Activity happened this week."],
            "attention": ["Someone said something. ^[S-not-in-payload @ nowhere]"],
            "by_account": [],
            "suggested_actions": [],
        }

    result, _reasons = _call(db, FakeLLMClient(_responder))
    assert result.attention == []
    assert result.this_week == ["Activity happened this week."]

    with db.cursor() as cur:
        cur.execute(
            "SELECT payload->>'detail' FROM review_queue "
            "WHERE payload->>'surface' = 'executive_summary_attention'"
        )
        rows = [r[0] for r in cur.fetchall()]
    assert any("unsupported citation" in r for r in rows), rows


def test_drops_a_bullet_with_an_unsupplied_numeral(db) -> None:
    citation, account_slug = _seed_account_claim(db)
    del account_slug

    def _responder(*, system, user, json_schema, model, surface):
        del user, json_schema, model, surface
        return {
            "this_week": [],
            "attention": [f"Acme Health has 47 open tickets. {citation}"],
            "by_account": [],
            "suggested_actions": [],
        }

    result, _reasons = _call(db, FakeLLMClient(_responder))
    assert result.attention == []

    with db.cursor() as cur:
        cur.execute(
            "SELECT payload->>'detail' FROM review_queue "
            "WHERE payload->>'surface' = 'executive_summary_attention'"
        )
        rows = [r[0] for r in cur.fetchall()]
    assert any("47" in r for r in rows), rows


def test_attention_bullet_citing_the_quiet_threshold_is_kept_but_a_wrong_count_is_dropped(
    db,
) -> None:
    """Coordinator follow-up (2026-09-18, after bughunt-2026-09-18-report.md
    findings 1/3): the day threshold a quiet-accounts bucket is measured
    against (30/60/90, `diff.QUIET_ACCOUNT_THRESHOLDS`) is now a named
    `NumberFacts` entry here too -- not just the bucket's own account
    COUNT (`quiet_30_count`, already a fact) -- since this summary is
    shown the full quiet-accounts list and is even more likely than the
    headline to narrate "quiet for 30 days". A bullet correctly citing
    that threshold must pass the gate; one inventing a different day
    count must still fail it."""
    citation, _account_slug = _seed_account_claim(db)

    def _accepted_responder(*, system, user, json_schema, model, surface):
        del system, json_schema, model, surface
        assert "quiet_threshold_30_days" in user
        return {
            "this_week": [],
            "attention": [f"Acme Health has gone quiet for 30 days. {citation}"],
            "by_account": [],
            "suggested_actions": [],
        }

    accepted, _reasons = _call(db, FakeLLMClient(_accepted_responder))
    assert accepted.attention == [f"Acme Health has gone quiet for 30 days. {citation}"]

    def _rejected_responder(*, system, user, json_schema, model, surface):
        del system, json_schema, model, surface
        return {
            "this_week": [],
            "attention": [f"Acme Health has gone quiet for 45 days. {citation}"],
            "by_account": [],
            "suggested_actions": [],
        }

    rejected, _reasons2 = _call(db, FakeLLMClient(_rejected_responder))
    assert rejected.attention == []


def test_drops_a_by_account_paragraph_naming_an_unknown_account(db) -> None:
    _seed_account_claim(db)

    def _responder(*, system, user, json_schema, model, surface):
        del user, json_schema, model, surface
        return {
            "this_week": [],
            "attention": [],
            "by_account": [
                {"account_slug": "not-a-real-active-account", "paragraph": "Made up account."}
            ],
            "suggested_actions": [],
        }

    result, _reasons = _call(db, FakeLLMClient(_responder))
    assert result.by_account == []
    assert result.failure_reason is not None

    with db.cursor() as cur:
        cur.execute(
            "SELECT payload->>'detail' FROM review_queue "
            "WHERE payload->>'surface' = 'executive_summary_by_account'"
        )
        rows = [r[0] for r in cur.fetchall()]
    assert any("not-a-real-active-account" in r for r in rows), rows


def test_returns_failure_reason_when_the_model_returns_nothing(db) -> None:
    _seed_account_claim(db)

    def _responder(*, system, user, json_schema, model, surface):
        del user, json_schema, model, surface
        return {"this_week": [], "attention": [], "by_account": [], "suggested_actions": []}

    result, reasons = _call(db, FakeLLMClient(_responder))
    assert result.this_week == []
    assert result.attention == []
    assert result.by_account == []
    assert result.suggested_actions == []
    assert result.failure_reason == "the model returned nothing in any section"
    assert reasons == ["the model returned nothing in any section"]


def test_model_call_failure_is_caught_and_explained(db) -> None:
    _seed_account_claim(db)

    class _RaisingLLM:
        def complete(self, **kwargs):
            del kwargs
            raise RuntimeError("provider unavailable")

    result, reasons = _call(db, _RaisingLLM())
    assert result.this_week == []
    assert "the model call failed" in (result.failure_reason or "")
    assert reasons and "the model call failed" in reasons[0]


def test_over_the_line_cap_rejects_the_whole_summary(db) -> None:
    citation, account_slug = _seed_account_claim(db)
    del account_slug

    def _responder(*, system, user, json_schema, model, surface):
        del user, json_schema, model, surface
        # Every one of these bullets is individually valid (cites the one
        # real marker, no fabricated numerals) -- only their sheer COUNT
        # should trip the cap.
        many_bullets = [
            f"Concern number restated. {citation}" for _ in range(MAX_RENDERED_LINES + 5)
        ]
        return {
            "this_week": [],
            "attention": many_bullets,
            "by_account": [],
            "suggested_actions": [],
        }

    result, reasons = _call(db, FakeLLMClient(_responder))
    assert result.attention == []
    assert result.failure_reason is not None
    assert "line cap" in result.failure_reason
    assert reasons and "line cap" in reasons[0]

    with db.cursor() as cur:
        cur.execute(
            "SELECT payload->>'detail' FROM review_queue "
            "WHERE payload->>'surface' = 'executive_summary'"
        )
        rows = [r[0] for r in cur.fetchall()]
    assert any("line cap" in r for r in rows), rows


def test_suggested_actions_can_name_the_owner_using_the_per_owner_open_count(db) -> None:
    """Owners, Phase 1 (01-spec/actions-and-owners-2026-09-19.md build
    item 7): "the number table ... gains ... per-owner counts" so
    "suggested actions name the owner" -- a bullet naming an owner's own
    open-action count (with NO citation of its own -- the count is a
    figure the report itself computed, not a claim) must survive the
    numeral gate. Before this feature, "2" was never a licensed number
    for this surface and the whole bullet would have been dropped."""
    diff, waves, _bare_actions, coverage, themes = _bare_feeds()
    actions = ActionsFeed(
        open=[
            ActionRow(
                id=1,
                note_slug="action-owner-count-1",
                what="Follow up on the delivery complaint",
                owner="Sam Okafor",
                owner_crm_id="sam-okafor",
                due=None,
                status="open",
            ),
            ActionRow(
                id=2,
                note_slug="action-owner-count-2",
                what="Send the renewal deck",
                owner="Sam Okafor",
                owner_crm_id="sam-okafor",
                due=None,
                status="open",
            ),
        ]
    )

    def _responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model, surface
        return {
            "this_week": [],
            "attention": [],
            "by_account": [],
            "suggested_actions": ["Sam Okafor has 2 open actions and should be checked in on."],
        }

    result = build_executive_summary(
        db,
        FakeLLMClient(_responder),
        since=SINCE,
        as_of=AS_OF,
        diff=diff,
        themes=themes,
        waves=waves,
        questions=[],
        outliers=[],
        contact_changes=[],
        actions=actions,
        coverage=coverage,
    )

    assert result.suggested_actions == [
        "Sam Okafor has 2 open actions and should be checked in on."
    ]
