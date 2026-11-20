"""Tests for the `research` CLI subcommand and its underlying pipeline
(`01-spec/actions-and-owners-2026-09-19.md`, build item 3): a single-
candidate harness for testing the action researcher alone -- find_
candidates -> filter -> researcher -> run_grounding_gate -> queue_
action_card, with NO report regeneration. Nine scratch acceptance runs
never produced a passing card because every iteration cost a full
`build_weekly` run (~20 minutes, 8 Sonnet prose calls) just to test this
one stage; this harness is the fix.

`test_report_helpers`'s `full_graph`/`report_llm` fixtures (the same
session-scoped, corpus-backed graph `test_report_weekly.py` already
builds `build_weekly` reports against for week 2026-W36) are imported
the SAME bare-module way `test_report_weekly.py` imports them --
importing them under a different qualified name (`tests.test_report_
helpers`) would give pytest a SECOND, distinct module object and a
second, competing session-scoped `full_graph` fixture instance racing to
initialize the same underlying database.
"""

# ruff: noqa: F811  (pytest fixtures imported by name; see test_report_helpers.py)

from __future__ import annotations

from datetime import timedelta

import pytest
from test_report_helpers import FullGraph, full_graph, report_llm  # noqa: F401

from portolan.report import weekly
from portolan.report.researcher import ActionCard, ResearchRunDiagnostics
from portolan.report.util import week_bounds
from portolan.report.weekly import UnknownCandidateSlugError
from tests.test_report_grounding import FakeSupportJudge

WEEK = "2026-W36"


# ---------------------------------------------------------------------
# Argparse wiring -- construction only, no DB (same discipline as
# test_report_weekly_cli.py's own `weekly` subcommand tests).
# ---------------------------------------------------------------------


def test_research_subcommand_parses_its_own_flags() -> None:
    args = weekly._build_arg_parser().parse_args(
        [
            "research",
            "--week",
            WEEK,
            "--clock",
            "world",
            "--embed",
            "fastembed",
            "--llm",
            "agent_sdk",
            "--candidate",
            "cand-1",
            "--candidate",
            "cand-2",
        ]
    )
    assert args.command == "research"
    assert args.week == WEEK
    assert args.clock == "world"
    assert args.embed == "fastembed"
    assert args.llm == "agent_sdk"
    assert args.candidate_slugs == ["cand-1", "cand-2"]
    assert args.all is False


def test_research_subcommand_defaults_clock_and_embed() -> None:
    args = weekly._build_arg_parser().parse_args(["research", "--week", WEEK, "--all"])
    assert args.clock == "record"
    assert args.embed == "fake"
    assert args.llm is None
    assert args.all is True
    assert args.candidate_slugs is None


def test_research_cli_refuses_neither_candidate_nor_all(capsys) -> None:
    args = weekly._build_arg_parser().parse_args(["research", "--week", WEEK])
    rc = weekly._run_research_cli(args)
    assert rc == 2
    assert "--candidate" in capsys.readouterr().err


def test_research_cli_refuses_both_candidate_and_all(capsys) -> None:
    args = weekly._build_arg_parser().parse_args(
        ["research", "--week", WEEK, "--all", "--candidate", "cand-1"]
    )
    rc = weekly._run_research_cli(args)
    assert rc == 2
    assert "--candidate" in capsys.readouterr().err


# ---------------------------------------------------------------------
# run_research_harness end to end, on the real corpus-backed test
# database, with a fake researcher and a fake judge -- no paid calls.
# ---------------------------------------------------------------------


class _FakeResearcherFromClaims:
    """A `Researcher` a test fully controls, but that still exercises the
    REAL mechanical/referential gate: for a candidate that carries at
    least one triggering claim id and an assigned vendor-side owner, it
    copies that claim's own stored `claim_text`/`quote`/`source_slug`/
    `locator` verbatim (the same "copy, never paraphrase" contract a real
    researcher session must follow) into a genuine `action` card.
    Candidates with no claim to cite, or no owner, get `None` -- the same
    "gave up" shape a real `no_card` reply produces."""

    def __init__(self, conn, *, as_of) -> None:
        self.conn = conn
        self.as_of = as_of
        self.diagnostics: list[ResearchRunDiagnostics] = []

    def run(self, candidate) -> ActionCard | None:
        if not candidate.claim_ids or not candidate.owner_crm_id:
            self.diagnostics.append(
                ResearchRunDiagnostics(
                    candidate_slug=candidate.slug,
                    tool_calls=1,
                    wall_time_s=0.1,
                    outcome="no_action",
                    detail="fake researcher: nothing to cite or no assigned owner",
                )
            )
            return None

        claim_id = candidate.claim_ids[0]
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT claim_text, quote, source_slug, locator FROM claims WHERE id = %s",
                (claim_id,),
            )
            row = cur.fetchone()
        if row is None:
            self.diagnostics.append(
                ResearchRunDiagnostics(
                    candidate_slug=candidate.slug,
                    tool_calls=1,
                    wall_time_s=0.1,
                    outcome="error",
                    detail="fake researcher: triggering claim vanished",
                )
            )
            return None
        claim_text, quote, source_slug, locator = row

        due = self.as_of + timedelta(days=7)
        card = ActionCard(
            run_id="fake-run",
            candidate_slug=candidate.slug,
            brief_version="fake-brief-v1",
            account=candidate.account_slug,
            verdict="action",
            what=f"Follow up on the {candidate.rule} signal.",
            evidence=[
                {
                    "claim_id": claim_id,
                    "text": claim_text,
                    "quote": quote,
                    "source_slug": source_slug,
                    "locator": locator,
                }
            ],
            findings=[],
            inference=[],
            owner=candidate.owner_crm_id,
            due=due,
            due_rule="next_report",
            theme=candidate.theme_slug,
            confidence="medium",
            confidence_refs={"evidence_refs": [0], "finding_refs": []},
        )
        self.diagnostics.append(
            ResearchRunDiagnostics(
                candidate_slug=candidate.slug, tool_calls=5, wall_time_s=2.5, outcome="card"
            )
        )
        return card


def _assign_a_vendor_side_owner(conn, *, account_slug: str) -> None:
    """`test_report_helpers.full_graph` runs the ingest pipeline only --
    it never runs Phase 1's owner assignment (`scripts/set_owners.py`),
    so every account in this corpus-backed test database starts with no
    `owner_entity_id` at all. Set one directly, the same effect Phase 1's
    real assignment has on the live graph, so a candidate on this account
    carries a real, vendor-side `owner_crm_id` the fake researcher (and
    the gate's owner check) can work with."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM registry_entities WHERE kind = 'internal_person' ORDER BY id LIMIT 1"
        )
        owner_row = cur.fetchone()
    assert owner_row is not None, "expected at least one vendor-side contact in the corpus"
    owner_id = owner_row[0]
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE registry_entities SET owner_entity_id = %s WHERE note_slug = %s",
            (owner_id, account_slug),
        )
    conn.commit()


def test_run_research_harness_end_to_end_with_a_fake_researcher_and_judge(
    full_graph: FullGraph, report_llm, capsys
) -> None:
    """Build item 3's own acceptance shape: find_candidates -> filter ->
    researcher -> gate -> queue, no report regeneration -- and the per-
    candidate line actually prints, with a fake researcher and a fake
    judge on the test database (no paid model call anywhere)."""
    _since, as_of = week_bounds(WEEK)
    researcher = _FakeResearcherFromClaims(full_graph.conn, as_of=as_of)
    judge = FakeSupportJudge(default="supported")

    # Probe run: discover a real candidate with a triggering claim to cite,
    # then give its account a real owner (see `_assign_a_vendor_side_owner`)
    # before the run this test actually asserts against -- `find_candidates`
    # reads `owner_entity_id` fresh every call, so re-running afterward
    # picks it up.
    probe_outcomes, _probe_themes = weekly.run_research_harness(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=WEEK,
        clock="record",
        connector=full_graph.connector,
        candidate_slugs=None,
        researcher=researcher,
        judge=judge,
    )
    citable = next(o for o in probe_outcomes if o.candidate.claim_ids)
    _assign_a_vendor_side_owner(full_graph.conn, account_slug=citable.candidate.account_slug)

    outcomes, themes = weekly.run_research_harness(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=WEEK,
        clock="record",
        connector=full_graph.connector,
        candidate_slugs=None,
        researcher=researcher,
        judge=judge,
    )

    assert outcomes, "expected at least one candidate for the corpus's own W36 window"
    assert themes is not None
    passing = [o for o in outcomes if o.card is not None and o.gate is not None and o.gate.passed]
    assert passing, "expected at least one candidate the fake researcher could ground"
    assert any(o.queued for o in passing)

    capsys.readouterr()  # drain anything printed by the feeds above
    weekly._print_research_results(outcomes)
    printed = capsys.readouterr().out

    queued_outcome = next(o for o in outcomes if o.queued)
    assert queued_outcome.candidate.slug in printed
    assert "gate=queued" in printed
    assert "verdict=action" in printed
    assert "submit_card_calls=" in printed
    assert "tool_calls=" in printed


def test_run_research_harness_filters_to_the_requested_candidate_slugs(
    full_graph: FullGraph, report_llm
) -> None:
    _since, as_of = week_bounds(WEEK)
    researcher = _FakeResearcherFromClaims(full_graph.conn, as_of=as_of)
    judge = FakeSupportJudge(default="supported")

    all_outcomes, _themes = weekly.run_research_harness(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=WEEK,
        clock="record",
        connector=full_graph.connector,
        candidate_slugs=None,
        researcher=researcher,
        judge=judge,
    )
    assert len(all_outcomes) >= 1
    one_slug = all_outcomes[0].candidate.slug

    filtered, _themes2 = weekly.run_research_harness(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=WEEK,
        clock="record",
        connector=full_graph.connector,
        candidate_slugs=[one_slug],
        researcher=researcher,
        judge=judge,
    )
    assert len(filtered) == 1
    assert filtered[0].candidate.slug == one_slug


def test_run_research_harness_unknown_candidate_slug_raises_with_valid_slugs_listed(
    full_graph: FullGraph, report_llm
) -> None:
    """S7 (`01-spec/reviews/review-action-researcher-2026-09-20.md`,
    should-fix 7, review point (l)): `research --candidate <typo>` used
    to pay for the full feed pass and then print "no candidates
    matched", exit 0 -- indistinguishable from a genuinely empty week.
    The feed pass genuinely cannot be skipped ahead of validating the
    slug (there is no cheaper way to know the valid slugs), but the
    harness now validates as soon as `find_candidates` returns, before
    spending anything further on a researcher session or a judge call,
    and raises naming both the bad slug and every real one this week's
    graph has."""
    _since, as_of = week_bounds(WEEK)
    researcher = _FakeResearcherFromClaims(full_graph.conn, as_of=as_of)
    judge = FakeSupportJudge(default="supported")

    with pytest.raises(UnknownCandidateSlugError) as excinfo:
        weekly.run_research_harness(
            full_graph.conn,
            report_llm,
            full_graph.embedder,
            week=WEEK,
            clock="record",
            connector=full_graph.connector,
            candidate_slugs=["not-a-real-candidate-slug"],
            researcher=researcher,
            judge=judge,
        )
    assert excinfo.value.unknown == ["not-a-real-candidate-slug"]
    assert excinfo.value.valid, "expected at least one real candidate slug for this week"
    assert "not-a-real-candidate-slug" not in excinfo.value.valid


def test_research_cli_exits_nonzero_and_prints_valid_slugs_for_an_unknown_candidate(
    db, db_url: str, monkeypatch, capsys
) -> None:
    """The CLI half of should-fix 7: `_run_research_cli` turns `Unknown
    CandidateSlugError` into a non-zero exit with the bad slug and every
    valid one printed to stderr -- `run_research_harness` itself is
    monkeypatched here (its own raise is proven directly above, against
    the real corpus) so this test only needs a migrated schema, not a
    full feed pass."""

    def _raising_harness(*args, **kwargs):
        del args, kwargs
        raise UnknownCandidateSlugError(unknown=["typo-slug"], valid=["real-slug-a", "real-slug-b"])

    monkeypatch.setattr(weekly, "run_research_harness", _raising_harness)

    args = weekly._build_arg_parser().parse_args(
        ["research", "--week", WEEK, "--llm", "fake", "--candidate", "typo-slug"]
    )
    rc = weekly._run_research_cli(args)

    assert rc == 2
    err = capsys.readouterr().err
    assert "typo-slug" in err
    assert "real-slug-a" in err
    assert "real-slug-b" in err
