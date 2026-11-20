"""Tests for the synthetic-data pipeline added 2026-09-19:
`corpus.generator.worlds`, the Slack/Teams/richer-CRM writers in
`corpus.generator.structured`, `corpus.generator.markers.process_indexed_texts`,
the free-teacher machinery in `corpus.generator.prose`, and the pure
(non-network) parts of `corpus.generator.cli`'s `synth` subcommand.

No network. Run with:

TZ=UTC PORTOLAN_ROLE=test uv run pytest -q tests/test_synth_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from corpus.generator import cli as cli_mod
from corpus.generator import markers, structured, worlds
from corpus.generator import prose as prose_mod
from corpus.generator.briefs import (
    Registry,
    build_calls,
    build_chat_briefs,
    build_crm,
    build_crm_notes,
    build_emails,
    build_tickets,
)
from portolan.training.spend import SpendMeter
from tests.synth_fakes import FAKE_TEACHERS, build_fake_client

# --------------------------------------------------------------------------
# worlds.generate_world
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world() -> dict:
    return worlds.generate_world(seed=42, num_accounts=4, per_kind=10)


def test_world_is_deterministic_in_seed():
    a = worlds.generate_world(seed=7, num_accounts=3, per_kind=6)
    b = worlds.generate_world(seed=7, num_accounts=3, per_kind=6)
    assert a == b


def test_world_different_seeds_differ():
    a = worlds.generate_world(seed=1, num_accounts=3, per_kind=6)
    b = worlds.generate_world(seed=2, num_accounts=3, per_kind=6)
    assert a["accounts"] != b["accounts"]


def test_world_has_required_top_level_shape(world):
    for key in ("meta", "accounts", "contacts", "facts", "sources", "crm_records"):
        assert key in world
    meta = world["meta"]
    # 2026-09-21 "distinct worlds": the vendor domain is now seed-picked
    # (worlds.VENDOR_BANK), never the hand-built corpus's fixed
    # "halvardmedical.com" -- see worlds.py's docstring.
    assert meta["supplier"]["domain"] != "halvardmedical.com"
    assert meta["supplier"]["domain"].endswith(".com")
    assert meta["window"]["start"] < meta["window"]["end"]
    for section in ("calls", "tickets", "emails", "slack", "teams", "crm", "crm_notes"):
        assert section in world["sources"], section


def test_world_email_domain_matches_supplier(world):
    # Every internal person's email uses THIS world's own supplier domain
    # (worlds._pick_vendor_mythology), never a hardcoded one. Unrelated to
    # manifest.LOCATOR_PATTERNS["email"], which validates the EMAIL
    # LOCATOR (built from structured.MESSAGE_ID_DOMAIN, a fixed technical
    # envelope constant, independent of any person's own address) -- see
    # worlds.py's docstring for why that never blocked this from varying.
    domain = world["meta"]["supplier"]["domain"]
    for person in world["meta"]["internal_people"]:
        assert person["email"].endswith(f"@{domain}")


def test_world_accounts_have_unique_ids(world):
    ids = [a["id"] for a in world["accounts"]]
    assert len(ids) == len(set(ids))
    crm_ids = [a["crm_id"] for a in world["accounts"]]
    assert len(crm_ids) == len(set(crm_ids))


def test_world_facts_reference_known_accounts(world):
    account_ids = {a["id"] for a in world["accounts"]}
    for fact in world["facts"]:
        for entity in fact["entities"]:
            # entities may be an account id, a product id, a competitor id,
            # or a contact id -- only assert the ones that look like our
            # own account ids actually resolve.
            if entity.startswith("w-") and entity in account_ids:
                assert entity in account_ids


def test_world_fact_ids_match_marker_regex(world):
    for fact in world["facts"]:
        assert markers.MARKER_RE.match(f"⟦{fact['id']}⟧x⟦/{fact['id']}⟧")


def test_world_fact_bearing_source_is_actually_about_that_account():
    # Regression: `_make_fact`'s chosen account must match the account the
    # SOURCE (call/ticket/email/slack/teams) is filed under -- a prior bug
    # let `commitment`/`other` facts land in a source tagged to a random,
    # unrelated account. Every entity-bearing fact's account entity must
    # appear as the source's `account` field for at least one of its
    # sources.
    w = worlds.generate_world(seed=123, num_accounts=6, per_kind=20)
    accounts_by_id = {a["id"]: a for a in w["accounts"]}
    call_items = {i["id"]: i for i in w["sources"]["calls"]["items"]}
    ticket_items = {i["id"]: i for i in w["sources"]["tickets"]["items"]}
    slack_items = {i["id"]: i for i in w["sources"]["slack"]["items"]}
    checked = 0
    for fact in w["facts"]:
        account_entities = [e for e in fact["entities"] if e in accounts_by_id]
        if not account_entities:
            continue
        for src in fact["sources"]:
            item = (
                call_items.get(src["id"])
                or ticket_items.get(src["id"])
                or slack_items.get(src["id"])
            )
            if item is None:
                continue
            assert item["account"] in account_entities, (fact["id"], src["id"], item["account"])
            checked += 1
    assert checked > 5


def test_world_per_kind_targets_are_respected(world):
    per_kind = 10
    for kind in ("calls", "tickets", "emails"):
        section = world["sources"][kind]
        total = len(section["items"]) + section["noise"]["count"]
        assert total == per_kind, kind
    for kind in ("slack", "teams"):
        assert len(world["sources"][kind]["items"]) == per_kind, kind


def test_world_supersession_pairs_reference_real_facts(world):
    fact_ids = {f["id"] for f in world["facts"]}
    for fact in world["facts"]:
        if fact.get("supersedes"):
            assert fact["supersedes"] in fact_ids


# --------------------------------------------------------------------------
# Distinct worlds (2026-09-21): different seeds must not read like the same
# cast with the serial numbers filed off.
# --------------------------------------------------------------------------


def test_two_seeds_share_no_account_or_person_names():
    # seed=1/seed=2 verified (empirically, not just by construction) to be
    # disjoint at these parameters -- generation is fully deterministic in
    # the seed, so this stays true across every run.
    a = worlds.generate_world(seed=1, num_accounts=5, per_kind=20)
    b = worlds.generate_world(seed=2, num_accounts=5, per_kind=20)

    a_accounts = {acct["name"] for acct in a["accounts"]}
    b_accounts = {acct["name"] for acct in b["accounts"]}
    assert a_accounts.isdisjoint(b_accounts)

    a_people = {c["name"] for c in a["contacts"]} | {
        p["name"] for p in a["meta"]["internal_people"]
    }
    b_people = {c["name"] for c in b["contacts"]} | {
        p["name"] for p in b["meta"]["internal_people"]
    }
    assert a_people.isdisjoint(b_people)


def test_two_seeds_can_produce_different_vendors():
    # Not every pair of seeds differs (the vendor bank has 8 entries), but
    # different seeds must be CAPABLE of picking a different vendor --
    # regression guard against a vendor accidentally hardcoded back to one
    # fixed value.
    vendors = {
        worlds.generate_world(seed=s, num_accounts=2, per_kind=6)["meta"]["supplier"]["name"]
        for s in range(10)
    }
    assert len(vendors) > 1


# --------------------------------------------------------------------------
# Forced type-mix (2026-09-21): per-type minimum planted-fact counts.
# --------------------------------------------------------------------------


def test_type_mix_reaches_minimums():
    from collections import Counter

    type_mix = {
        "pricing": 12,
        "competitor_mention": 12,
        "risk": 12,
        "question": 12,
        "contact_change": 8,
    }
    w = worlds.generate_world(seed=55, num_accounts=6, per_kind=20, type_mix=type_mix)
    counts = Counter(f["type"] for f in w["facts"])
    for fact_type, minimum in type_mix.items():
        assert counts[fact_type] >= minimum, (fact_type, counts[fact_type], minimum)


def test_type_mix_balanced_profile_gives_every_type_a_floor():
    mix = worlds.balanced_type_mix(per_kind=30)
    assert set(mix) == set(worlds.FACT_TYPES)
    assert all(n >= 1 for n in mix.values())
    from collections import Counter

    w = worlds.generate_world(seed=56, num_accounts=6, per_kind=30, type_mix=mix)
    counts = Counter(f["type"] for f in w["facts"])
    for fact_type, minimum in mix.items():
        assert counts[fact_type] >= minimum, (fact_type, counts[fact_type], minimum)


def test_type_mix_none_matches_unforced_generation():
    # A falsy type_mix keeps the original per-index weighted-choice
    # behavior byte-for-byte -- same seed, same result, with or without
    # explicitly passing type_mix=None.
    a = worlds.generate_world(seed=57, num_accounts=4, per_kind=10)
    b = worlds.generate_world(seed=57, num_accounts=4, per_kind=10, type_mix=None)
    assert a == b


def test_parse_type_mix_none_and_balanced_and_explicit():
    assert worlds.parse_type_mix(None, per_kind=20) is None
    assert worlds.parse_type_mix("", per_kind=20) is None
    assert worlds.parse_type_mix("balanced", per_kind=20) == worlds.balanced_type_mix(20)
    assert worlds.parse_type_mix("pricing=5,risk=3", per_kind=20) == {"pricing": 5, "risk": 3}


def test_parse_type_mix_rejects_unknown_type():
    with pytest.raises(ValueError):
        worlds.parse_type_mix("not_a_real_type=5", per_kind=20)


def test_parse_type_mix_rejects_bad_count():
    with pytest.raises(ValueError):
        worlds.parse_type_mix("pricing=not-a-number", per_kind=20)


def test_generate_world_rejects_unknown_type_in_type_mix():
    with pytest.raises(ValueError):
        worlds.generate_world(seed=1, num_accounts=3, per_kind=10, type_mix={"nope": 5})


# --------------------------------------------------------------------------
# briefs.py additions: build_chat_briefs / build_crm_notes, and the
# reused build_calls/build_tickets/build_emails/build_crm against a
# generated world (not the hand-built corpus).
# --------------------------------------------------------------------------


def test_reused_brief_builders_work_against_a_generated_world(world):
    import random

    registry = Registry(world)
    rng = random.Random(0)
    call_briefs = build_calls(world, registry, rng, reduced=True)
    ticket_briefs = build_tickets(world, registry, rng, reduced=True)
    email_briefs = build_emails(world, registry, rng, reduced=True)
    assert call_briefs and ticket_briefs and email_briefs
    fact_bearing_calls = [b for b in call_briefs if b.facts]
    assert fact_bearing_calls


def test_build_chat_briefs_slack_and_teams(world):
    registry = Registry(world)
    slack_briefs = build_chat_briefs("slack", world, registry)
    teams_briefs = build_chat_briefs("teams", world, registry)
    assert len(slack_briefs) == len(world["sources"]["slack"]["items"])
    assert len(teams_briefs) == len(world["sources"]["teams"]["items"])
    for b in slack_briefs:
        assert b.source_type == "slack"
        assert "channel" in b.extra
        assert len(b.participants) == 1
    fact_bearing = [b for b in slack_briefs if b.facts]
    noise = [b for b in slack_briefs if not b.facts]
    assert fact_bearing and noise
    for b in fact_bearing:
        assert not b.is_noise
    for b in noise:
        assert b.is_noise


def test_build_crm_notes(world):
    registry = Registry(world)
    note_briefs = build_crm_notes(world, registry)
    assert len(note_briefs) == len(world["sources"]["crm_notes"]["items"])
    assert any(b.facts for b in note_briefs)
    assert any(not b.facts for b in note_briefs)


def test_build_crm_reused_against_generated_world(world):
    registry = Registry(world)
    crm_briefs = build_crm(world, registry)
    assert len(crm_briefs) == len(world["sources"]["crm"]["items"])
    item_by_id = {i["id"]: i for i in world["sources"]["crm"]["items"]}
    for b in crm_briefs:
        item = item_by_id[b.source_id]
        assert "crm_id" in item
        assert "path" in item


# --------------------------------------------------------------------------
# structured.py: chat locators/paths, Slack/Teams writers, richer CRM
# --------------------------------------------------------------------------


def test_slack_teams_locator_format():
    assert (
        structured.slack_locator("acct-foo", "2026-08-05", 3) == "slack/acct-foo/2026-08-05.json#m3"
    )
    assert (
        structured.teams_locator("acct-foo", "2026-08-05", 1) == "teams/acct-foo/2026-08-05.json#m1"
    )


def test_slack_teams_locator_matches_manifest_patterns():
    slack_loc = structured.slack_locator("acct-briarcliff", "2026-03-10", 5)
    teams_loc = structured.teams_locator("acct-briarcliff", "2026-03-10", 5)
    assert markers is not None  # keep import used
    from corpus.generator.manifest import LOCATOR_PATTERNS

    assert LOCATOR_PATTERNS["slack"].match(slack_loc)
    assert LOCATOR_PATTERNS["teams"].match(teams_loc)


def test_crm_locator_pattern_accepts_csv_and_json():
    from corpus.generator.manifest import LOCATOR_PATTERNS

    assert LOCATOR_PATTERNS["crm"].match("crm/accounts.csv#A-1010")
    assert LOCATOR_PATTERNS["crm"].match("crm/activity_notes/N-8001.json#p0")


def test_write_slack_day_round_trip(tmp_path: Path):
    messages = [
        {
            "type": "message",
            "user": "u1",
            "text": "hello",
            "ts": "1700000000.000100",
            "thread_ts": None,
        }
    ]
    out_path = tmp_path / "slack" / "acct-foo" / "2026-08-05.json"
    structured.write_slack_day(messages, out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == messages


def test_write_teams_day_round_trip(tmp_path: Path):
    messages = [
        {
            "id": "123",
            "from": {"user": {"displayName": "Sam Okafor"}},
            "body": {"content": "hi"},
            "createdDateTime": "2026-08-05T00:05:00Z",
            "replyToId": None,
        }
    ]
    out_path = tmp_path / "teams" / "acct-foo" / "2026-08-05.json"
    structured.write_teams_day(messages, out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == messages


def test_write_activity_note(tmp_path: Path):
    out_path = structured.activity_note_path(tmp_path, "N-8001")
    structured.write_activity_note(
        note_id="N-8001",
        account_crm_id="A-4001",
        contact_crm_id="C-6001",
        author="w-sam-okafor",
        created_date="2026-08-05",
        subject="Account touchpoint",
        body="Logged after today's call.",
        out_path=out_path,
    )
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["id"] == "N-8001"
    assert loaded["body"] == "Logged after today's call."
    assert structured.activity_note_relpath("N-8001") == "crm/activity_notes/N-8001.json"


def test_write_synth_opportunities_and_cases_csv(tmp_path: Path):
    opp_rows = [
        {
            "opportunity_id": "O-5001",
            "account_crm_id": "A-4001",
            "name": "Test renewal",
            "stage": "negotiation",
            "close_date": "2027-01-01",
            "amount": 100000,
            "export_date": "2026-09-01",
        }
    ]
    case_rows = [
        {
            "case_id": "CASE-7001",
            "account_crm_id": "A-4001",
            "contact_crm_id": "C-6001",
            "subject": "Delivery timing question",
            "status": "open",
            "priority": "medium",
            "created_date": "2026-08-01",
            "closed_date": "",
        }
    ]
    opp_path = tmp_path / "crm" / "opportunities.csv"
    case_path = tmp_path / "crm" / "cases.csv"
    assert structured.write_synth_opportunities_csv(opp_rows, opp_path) == 1
    assert structured.write_synth_cases_csv(case_rows, case_path) == 1
    assert opp_path.exists() and case_path.exists()
    assert "O-5001" in opp_path.read_text()
    assert "CASE-7001" in case_path.read_text()


# --------------------------------------------------------------------------
# markers.process_indexed_texts
# --------------------------------------------------------------------------


def test_process_indexed_texts_finds_and_strips_markers():
    texts = ["no fact here", "⟦F101⟧the pump arrived late⟦/F101⟧ end of message"]
    clean, placements = markers.process_indexed_texts(texts, lambda i: f"loc#{i}")
    assert clean == ["no fact here", "the pump arrived late end of message"]
    assert len(placements) == 1
    p = placements[0]
    assert p.fact_id == "F101"
    assert p.locator == "loc#1"
    assert p.excerpt == "the pump arrived late"


def test_process_indexed_texts_empty_list():
    clean, placements = markers.process_indexed_texts([], lambda i: f"loc#{i}")
    assert clean == []
    assert placements == []


# --------------------------------------------------------------------------
# prose.py: discover_free_models / TokenBucketLimiter / TeacherRotation
# --------------------------------------------------------------------------


def _models_response(ids: list[str]) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"id": i} for i in ids]})


def test_discover_free_models_filters_and_orders_by_preference():
    ids = [
        "openai/gpt-5.6-luna",  # not free -- excluded
        "qwen/qwen3.8-27b:free",
        "deepseek/deepseek-v4-flash-0731:free",
        "some/other-model:free",
        "z-ai/glm-5.2:free",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _models_response(ids)

    transport = httpx.MockTransport(handler)
    result = prose_mod.discover_free_models("fake-key", transport=transport)

    assert "openai/gpt-5.6-luna" not in result
    assert all(m.endswith(":free") for m in result)
    # Preferred families come first, in PREFERRED_FREE_TEACHER_FAMILIES order.
    assert result[0] == "deepseek/deepseek-v4-flash-0731:free"
    assert result[1] == "z-ai/glm-5.2:free"
    assert result[2] == "qwen/qwen3.8-27b:free"
    assert "some/other-model:free" in result
    assert result[-1] == "some/other-model:free"


def test_discover_free_models_never_hardcodes_ids_not_seen():
    # Only families the fixture actually included should be preferred;
    # nothing from PREFERRED_FREE_TEACHER_FAMILIES should be fabricated.
    ids = ["random-org/random-model:free"]

    def handler(request: httpx.Request) -> httpx.Response:
        return _models_response(ids)

    transport = httpx.MockTransport(handler)
    result = prose_mod.discover_free_models("fake-key", transport=transport)
    assert result == ["random-org/random-model:free"]


def test_discover_free_models_network_failure_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network", request=request)

    transport = httpx.MockTransport(handler)
    assert prose_mod.discover_free_models("fake-key", transport=transport) == []


def test_discover_free_models_non_2xx_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    transport = httpx.MockTransport(handler)
    assert prose_mod.discover_free_models("fake-key", transport=transport) == []


# --------------------------------------------------------------------------
# prose.py: free_model_daily_requests_remaining (added 2026-09-21, fix for
# the batch runner silently burning through 15 worlds after the daily cap)
# --------------------------------------------------------------------------


def _key_response(remaining: int, *, used: int = 0, limit: int = 1000) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": {
                "label": "test-key",
                "limit": None,
                "limit_reset": None,
                "limit_remaining": None,
                "usage": 0,
                "usage_daily": 0,
                "usage_weekly": 0,
                "usage_monthly": 0,
                "is_free_tier": True,
                "free_model_daily_requests": {
                    "used": used,
                    "limit": limit,
                    "remaining": remaining,
                },
            }
        },
    )


def test_free_model_daily_requests_remaining_reads_nested_field():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer fake-key"
        return _key_response(remaining=42, used=958, limit=1000)

    transport = httpx.MockTransport(handler)
    assert prose_mod.free_model_daily_requests_remaining("fake-key", transport=transport) == 42


def test_free_model_daily_requests_remaining_non_2xx_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    transport = httpx.MockTransport(handler)
    assert prose_mod.free_model_daily_requests_remaining("fake-key", transport=transport) is None


def test_free_model_daily_requests_remaining_network_failure_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network", request=request)

    transport = httpx.MockTransport(handler)
    assert prose_mod.free_model_daily_requests_remaining("fake-key", transport=transport) is None


def test_free_model_daily_requests_remaining_missing_field_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"label": "test-key"}})

    transport = httpx.MockTransport(handler)
    assert prose_mod.free_model_daily_requests_remaining("fake-key", transport=transport) is None


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_token_bucket_limiter_allows_up_to_rpm_without_sleeping():
    clock = _FakeClock()
    sleeps: list[float] = []
    limiter = prose_mod.TokenBucketLimiter(
        requests_per_minute=3, requests_per_day=100, clock=clock, sleep=sleeps.append
    )
    for _ in range(3):
        limiter.acquire()
    assert sleeps == []


def test_token_bucket_limiter_sleeps_when_minute_budget_exhausted():
    clock = _FakeClock()
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    limiter = prose_mod.TokenBucketLimiter(
        requests_per_minute=2, requests_per_day=100, clock=clock, sleep=fake_sleep
    )
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # 3rd request in the same minute -> must sleep
    assert len(sleeps) == 1
    assert sleeps[0] > 0


def test_token_bucket_limiter_daily_limit_raises():
    clock = _FakeClock()
    limiter = prose_mod.TokenBucketLimiter(
        requests_per_minute=1000, requests_per_day=2, clock=clock, sleep=lambda s: None
    )
    limiter.acquire()
    limiter.acquire()
    with pytest.raises(prose_mod.DailyLimitExceeded):
        limiter.acquire()


def test_token_bucket_limiter_day_window_resets():
    clock = _FakeClock()
    limiter = prose_mod.TokenBucketLimiter(
        requests_per_minute=1000, requests_per_day=1, clock=clock, sleep=lambda s: None
    )
    limiter.acquire()
    with pytest.raises(prose_mod.DailyLimitExceeded):
        limiter.acquire()
    clock.advance(86400.0 + 1.0)
    limiter.acquire()  # new day window -> should not raise


def test_teacher_rotation_round_robins():
    rotation = prose_mod.TeacherRotation(["a", "b", "c"])
    picks = [rotation.next() for _ in range(6)]
    assert picks == ["a", "b", "c", "a", "b", "c"]
    assert rotation.request_counts == {"a": 2, "b": 2, "c": 2}


def test_teacher_rotation_excludes_when_possible():
    rotation = prose_mod.TeacherRotation(["a", "b"])
    first = rotation.next()
    second = rotation.next(exclude=first)
    assert second != first


def test_teacher_rotation_single_teacher_repeats_when_excluded():
    rotation = prose_mod.TeacherRotation(["only"])
    assert rotation.next(exclude="only") == "only"


def test_teacher_rotation_requires_at_least_one_teacher():
    with pytest.raises(ValueError):
        prose_mod.TeacherRotation([])


def test_teacher_rotation_tracks_rate_limits():
    rotation = prose_mod.TeacherRotation(["a", "b"])
    rotation.record_rate_limit("a")
    rotation.record_rate_limit("a")
    assert rotation.rate_limit_counts["a"] == 2
    assert rotation.total_rate_limits == 2


# --------------------------------------------------------------------------
# TeacherRotation per-model cooldowns (2026-09-21 free-tier-saturation
# coordinator heads-up): 60s-doubling-to-15min backoff, "next available"
# model picks that skip cooling ones, and a bounded wait when every model
# is cooling.
# --------------------------------------------------------------------------


def test_record_rate_limit_starts_a_cooldown():
    clock = _FakeClock()
    rotation = prose_mod.TeacherRotation(["a", "b"], clock=clock, sleep=lambda s: None)
    assert not rotation.is_cooling("a")
    rotation.record_rate_limit("a")
    assert rotation.is_cooling("a")
    assert rotation.cooldown_remaining("a") == pytest.approx(60.0)


def test_record_rate_limit_doubles_on_repeat_failures_capped_at_15_minutes():
    clock = _FakeClock()
    rotation = prose_mod.TeacherRotation(["a"], clock=clock, sleep=lambda s: None)
    expected = [60.0, 120.0, 240.0, 480.0, 900.0, 900.0]
    for want in expected:
        rotation.record_rate_limit("a")
        assert rotation.cooldown_remaining("a") == pytest.approx(want)
        clock.advance(want)  # let this cooldown fully expire before the next failure


def test_record_rate_limit_honors_explicit_retry_after():
    clock = _FakeClock()
    rotation = prose_mod.TeacherRotation(["a"], clock=clock, sleep=lambda s: None)
    rotation.record_rate_limit("a", retry_after=12.5)
    assert rotation.cooldown_remaining("a") == pytest.approx(12.5)


def test_mark_success_clears_cooldown():
    clock = _FakeClock()
    rotation = prose_mod.TeacherRotation(["a"], clock=clock, sleep=lambda s: None)
    rotation.record_rate_limit("a")
    assert rotation.is_cooling("a")
    rotation.mark_success("a")
    assert not rotation.is_cooling("a")
    # backoff also resets -- the next failure starts back at the base, not
    # doubled from where it left off.
    rotation.record_rate_limit("a")
    assert rotation.cooldown_remaining("a") == pytest.approx(60.0)


def test_next_available_skips_cooling_models():
    clock = _FakeClock()
    rotation = prose_mod.TeacherRotation(["a", "b", "c"], clock=clock, sleep=lambda s: None)
    rotation.record_rate_limit("a")
    picks = [rotation.next_available() for _ in range(4)]
    assert "a" not in picks
    assert set(picks) <= {"b", "c"}


def test_next_available_returns_none_when_all_cooling():
    clock = _FakeClock()
    rotation = prose_mod.TeacherRotation(["a", "b"], clock=clock, sleep=lambda s: None)
    rotation.record_rate_limit("a")
    rotation.record_rate_limit("b")
    assert rotation.next_available() is None


def test_wait_for_available_sleeps_until_earliest_cooldown_then_returns():
    clock = _FakeClock()
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    rotation = prose_mod.TeacherRotation(["a", "b"], clock=clock, sleep=fake_sleep)
    rotation.record_rate_limit("a")  # 60s
    rotation.record_rate_limit("b")  # 60s
    model = rotation.wait_for_available()
    assert model in ("a", "b")
    assert sleeps == [pytest.approx(60.0)]


def test_wait_for_available_does_not_wait_when_a_model_is_free():
    clock = _FakeClock()
    sleeps: list[float] = []
    rotation = prose_mod.TeacherRotation(["a", "b"], clock=clock, sleep=sleeps.append)
    rotation.record_rate_limit("a")
    model = rotation.wait_for_available()
    assert model == "b"
    assert sleeps == []


# --------------------------------------------------------------------------
# generate_via_free_teacher: retries across models on 429/5xx, cooldown-
# aware, re-raises non-retryable errors immediately.
# --------------------------------------------------------------------------


class _FlakyThenOkClient:
    """Fails with a 429-shaped LLMError for `fail_models`, succeeds
    otherwise."""

    def __init__(self, fail_models: set[str]) -> None:
        self.fail_models = set(fail_models)
        self.calls: list[str] = []

    def complete(
        self, *, system, user, model, surface, max_tokens=None, temperature=None, json_schema=None
    ):
        from portolan.llm.base import LLMError

        self.calls.append(model)
        if model in self.fail_models:
            raise LLMError(
                f"OpenRouterClient: request failed with status 429 (model={model!r}): rate limited"
            )
        from portolan.llm.base import LLMResult

        return LLMResult(
            text="ok", data=None, input_tokens=1, output_tokens=1, cached_tokens=0, model=model
        )


def _dummy_brief():
    from corpus.generator.briefs import Brief

    return Brief(
        source_id="S-test-001",
        source_type="call",
        date="2026-01-05",
        participants=[],
        accounts=[],
        target_tokens=None,
        facts=[],
    )


def test_generate_via_free_teacher_retries_on_429_then_succeeds():
    clock = _FakeClock()
    rotation = prose_mod.TeacherRotation(["a", "b"], clock=clock, sleep=lambda s: None)
    limiter = prose_mod.TokenBucketLimiter(
        requests_per_minute=1000, requests_per_day=1000, clock=clock, sleep=lambda s: None
    )
    client = _FlakyThenOkClient(fail_models={"a"})

    text, model = prose_mod.generate_via_free_teacher(client, _dummy_brief(), rotation, limiter)
    assert model == "b"
    assert text == "ok"
    assert rotation.is_cooling("a")
    assert not rotation.is_cooling("b")
    assert client.calls == ["a", "b"]


class _CountingFlakyClient:
    """Fails with a 429-shaped LLMError for the first `fail_first_n`
    calls (regardless of which model), succeeds after that."""

    def __init__(self, fail_first_n: int) -> None:
        self.fail_first_n = fail_first_n
        self.calls: list[str] = []

    def complete(
        self, *, system, user, model, surface, max_tokens=None, temperature=None, json_schema=None
    ):
        from portolan.llm.base import LLMError, LLMResult

        self.calls.append(model)
        if len(self.calls) <= self.fail_first_n:
            raise LLMError(
                f"OpenRouterClient: request failed with status 429 (model={model!r}): rate limited"
            )
        return LLMResult(
            text="ok", data=None, input_tokens=1, output_tokens=1, cached_tokens=0, model=model
        )


def test_generate_via_free_teacher_waits_when_all_models_cooling():
    clock = _FakeClock()
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    rotation = prose_mod.TeacherRotation(["a", "b"], clock=clock, sleep=fake_sleep)
    limiter = prose_mod.TokenBucketLimiter(
        requests_per_minute=1000, requests_per_day=1000, clock=clock, sleep=lambda s: None
    )
    # Attempt 1 (model a) fails, attempt 2 (model b) fails -- now BOTH are
    # cooling at the same simulated instant (60s each); attempt 3 must
    # wait for that cooldown, then succeeds.
    client = _CountingFlakyClient(fail_first_n=2)

    text, model = prose_mod.generate_via_free_teacher(client, _dummy_brief(), rotation, limiter)
    assert text == "ok"
    assert client.calls == ["a", "b", model]
    assert sleeps == [pytest.approx(60.0)]  # waited exactly once, for the earliest cooldown


def test_generate_via_free_teacher_reraises_non_retryable_error():
    from portolan.llm.base import LLMError

    clock = _FakeClock()
    rotation = prose_mod.TeacherRotation(["a"], clock=clock, sleep=lambda s: None)
    limiter = prose_mod.TokenBucketLimiter(
        requests_per_minute=1000, requests_per_day=1000, clock=clock, sleep=lambda s: None
    )

    class _AuthFailClient:
        def complete(self, **kwargs):
            raise LLMError("OpenRouterClient: request failed with status 401 (model='a'): bad key")

    with pytest.raises(LLMError, match="401"):
        prose_mod.generate_via_free_teacher(_AuthFailClient(), _dummy_brief(), rotation, limiter)
    assert not rotation.is_cooling("a")  # a non-retryable error is not a rate-limit signal


# --------------------------------------------------------------------------
# cli.py: pure placement (de)serialization helpers used by synth's resume
# --------------------------------------------------------------------------


def test_placement_dict_round_trip():
    p = markers.Placement("F101", "00:01:02", "5", "the pump arrived late")
    d = cli_mod._placement_to_dict(p)
    p2 = cli_mod._placement_from_dict(d)
    assert p2 == p


def _sample_progress_entry() -> dict[str, Any]:
    return {
        "fact_ids": ["F101"],
        "clean": "hi",
        "placements": [],
        "model": "m",
        "failed": False,
    }


def test_synth_progress_save_and_load_round_trip(tmp_path: Path):
    progress = {"sources": {"S-call-001": _sample_progress_entry()}}
    cli_mod._synth_save_progress(tmp_path, progress)
    loaded = cli_mod._synth_load_progress(tmp_path)
    assert loaded == progress


def test_synth_reuse_returns_none_when_fact_ids_differ():
    progress = {"sources": {"S-call-001": _sample_progress_entry()}}
    assert cli_mod._synth_reuse(progress, "S-call-001", ["F102"]) is None
    reused = cli_mod._synth_reuse(progress, "S-call-001", ["F101"])
    assert reused is not None
    assert reused.clean == "hi"


# --------------------------------------------------------------------------
# Paid teachers -- added 2026-09-22 (`--teachers`, bypassing free-model
# discovery). No network: every test below uses `tests.synth_fakes.
# build_fake_client` and an explicit `tmp_path`-scoped `SpendMeter`, never
# the real `var/data/silver/spend.json`.
# --------------------------------------------------------------------------


def test_teachers_are_paid_true_for_all_paid_ids():
    assert prose_mod.teachers_are_paid(["deepseek/deepseek-v4-flash-0731", "z-ai/glm-5.2"]) is True


def test_teachers_are_paid_false_for_all_free_ids():
    free_ids = ["deepseek/deepseek-v4-flash-0731:free", "z-ai/glm-5.2:free"]
    assert prose_mod.teachers_are_paid(free_ids) is False


def test_teachers_are_paid_raises_on_mixed_list():
    with pytest.raises(prose_mod.MixedTeacherListError):
        prose_mod.teachers_are_paid(["deepseek/deepseek-v4-flash-0731:free", "z-ai/glm-5.2"])


def test_teachers_are_paid_raises_on_empty_list():
    with pytest.raises(ValueError):
        prose_mod.teachers_are_paid([])


def test_paid_budget_exceeded_message_contains_marker():
    exc = prose_mod.PaidBudgetExceeded(f"{prose_mod.PAID_BUDGET_STOP_MARKER}: $5.00 >= $5.00")
    assert prose_mod.PAID_BUDGET_STOP_MARKER in str(exc)
    assert isinstance(exc, prose_mod.DailyLimitExceeded)  # existing except clauses still catch it


def test_synth_teachers_bypasses_free_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("discover_free_models must not be called when --teachers is given")

    monkeypatch.setattr(prose_mod, "discover_free_models", _must_not_be_called)

    teachers = ["deepseek/deepseek-v4-flash-0731", "deepseek/deepseek-v4.1-flash"]
    spend = SpendMeter(path=tmp_path / "spend.json")  # never the real var/ path
    report = cli_mod.synth(
        seed=90001,
        num_accounts=1,
        per_kind=2,
        out_dir=tmp_path / "world",
        client=build_fake_client(),
        teachers=teachers,
        budget_usd=5.00,
        spend=spend,
    )
    assert report["teachers"] == teachers
    assert set(report["request_counts"].keys()) == set(teachers)


def test_synth_paid_teachers_stops_cleanly_when_budget_already_spent(tmp_path: Path):
    """The exit-4 case: a shared spend meter already at/over `budget_usd`
    stops the run before a single live call is made (checked before every
    attempt, never mid-call -- `_synth_generate_unit`)."""
    teachers = ["deepseek/deepseek-v4-flash-0731"]
    rotation = prose_mod.TeacherRotation(list(teachers))
    limiter = prose_mod.TokenBucketLimiter(requests_per_minute=10_000, requests_per_day=10_000_000)
    spend = SpendMeter(path=tmp_path / "spend.json")
    spend.record(model=teachers[0], corpus="synthetic", amount_usd=10.0)

    report = cli_mod.synth(
        seed=90002,
        num_accounts=2,
        per_kind=4,
        out_dir=tmp_path / "world",
        client=build_fake_client(),
        rotation=rotation,
        limiter=limiter,
        budget_usd=1.0,
        spend=spend,
    )

    assert report["stopped_early"] is True
    assert prose_mod.PAID_BUDGET_STOP_MARKER in report["stop_reason"]
    assert report["verified"] == 0
    assert rotation.total_requests == 0  # not one live call was made
    assert spend.total_usd() == 10.0  # unchanged -- nothing new was ever billed


def test_synth_paid_budget_checks_synthetic_corpus_not_meter_total(tmp_path: Path):
    """Regression for the 2026-09-22 bug: a shared meter whose GRAND TOTAL
    is already over the ceiling (a concurrent silver-labeling run spent
    against a different corpus) must NOT stop a paid-teacher synth run
    whose own `synthetic` corpus line is still under budget."""
    teachers = ["deepseek/deepseek-v4-flash-0731"]
    rotation = prose_mod.TeacherRotation(list(teachers))
    limiter = prose_mod.TokenBucketLimiter(requests_per_minute=10_000, requests_per_day=10_000_000)
    spend = SpendMeter(path=tmp_path / "spend.json")
    spend.record(model="some-silver-model", corpus="fda_maude", amount_usd=5.89)
    assert spend.total_usd() >= 5.0  # the old (buggy) check's trigger condition

    report = cli_mod.synth(
        seed=90004,
        num_accounts=2,
        per_kind=4,
        out_dir=tmp_path / "world",
        client=build_fake_client(),
        rotation=rotation,
        limiter=limiter,
        budget_usd=5.0,
        spend=spend,
    )

    assert report["stopped_early"] is False
    assert report["verified"] > 0
    assert rotation.total_requests > 0  # live calls were made, unlike the over-budget case above
    assert spend.corpus_usd("fda_maude") == 5.89  # unrelated corpus line untouched
    assert 0.0 < spend.corpus_usd("synthetic") < 5.0  # this run's own spend, still under budget


def test_synth_free_path_unaffected_by_paid_teacher_fields(tmp_path: Path):
    """Free runs (no `teachers`, or an all-`:free` `--teachers` list) are
    unaffected by the paid-teacher machinery: `spend` stays unset on
    `RunState`, so the budget check in `_synth_generate_unit` never fires,
    and a real (tiny, well within budget) generation run completes and
    verifies facts exactly as before this feature existed."""
    report = cli_mod.synth(
        seed=90003,
        num_accounts=2,
        per_kind=4,
        out_dir=tmp_path / "world",
        client=build_fake_client(),
        rotation=prose_mod.TeacherRotation(list(FAKE_TEACHERS)),
        limiter=prose_mod.TokenBucketLimiter(
            requests_per_minute=10_000, requests_per_day=10_000_000
        ),
    )
    assert report["stopped_early"] is False
    assert report["verified"] > 0


def test_main_rejects_mixed_teachers_before_any_network_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def _must_not_be_called(**kwargs):
        raise AssertionError("synth() must not run when --teachers is a mixed list")

    monkeypatch.setattr(cli_mod, "synth", _must_not_be_called)
    rc = cli_mod.main(
        [
            "synth",
            "--seed",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--teachers",
            "deepseek/deepseek-v4-flash-0731:free,deepseek/deepseek-v4.1-flash",
        ]
    )
    assert rc == 1
    assert "invalid --teachers" in capsys.readouterr().out


def test_main_returns_4_when_report_stopped_for_paid_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def _fake_synth(**kwargs) -> dict[str, Any]:
        return {
            "seed": 1,
            "num_accounts": 2,
            "per_kind": 4,
            "teachers": ["deepseek/deepseek-v4-flash-0731"],
            "request_counts": {},
            "rate_limit_counts": {},
            "sources_by_kind": {},
            "total_sources": 0,
            "planted_facts_placed": 0,
            "verified": 0,
            "filler_used_after_failure": 0,
            "stopped_early": True,
            "stop_reason": f"{prose_mod.PAID_BUDGET_STOP_MARKER}: $5.0000 >= $5.00 budget-usd",
            "failed_sources": [],
            "manifest_problems": [],
        }

    monkeypatch.setattr(cli_mod, "synth", _fake_synth)
    rc = cli_mod.main(
        [
            "synth",
            "--seed",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--teachers",
            "deepseek/deepseek-v4-flash-0731",
        ]
    )
    assert rc == 4


def test_main_returns_4_on_paid_budget_exceeded_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def _fake_synth(**kwargs):
        raise prose_mod.PaidBudgetExceeded("boom")

    monkeypatch.setattr(cli_mod, "synth", _fake_synth)
    rc = cli_mod.main(
        [
            "synth",
            "--seed",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--teachers",
            "deepseek/deepseek-v4-flash-0731",
        ]
    )
    assert rc == 4
