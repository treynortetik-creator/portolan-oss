"""Tests for `corpus.generator.training_export` (added 2026-09-21).

Two layers:
  - Unit tests against a hand-built batch directory (manifest.json +
    truth-table.yaml + the source files themselves) for exact control over
    the verified/failed/noise distinctions `_should_export` has to get
    right -- no network, no real `synth()` run.
  - One end-to-end smoke test running the real `cli.synth()` orchestration
    against a fake, no-network teacher (`tests.synth_fakes`) and exporting
    the result, checking the record shapes it actually produces.

Run with:
    TZ=UTC PORTOLAN_ROLE=test uv run pytest -q tests/test_training_export.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from corpus.generator import training_export
from tests.synth_fakes import run_fake_synth

# --------------------------------------------------------------------------
# Hand-built batch fixture
# --------------------------------------------------------------------------


def _source(
    *,
    id: str,
    type: str,
    path: str,
    date: str = "2026-01-05",
    facts: list[dict[str, Any]] | None = None,
    is_noise: bool = False,
    model: str | None = "faketeacher/alpha:free",
) -> dict[str, Any]:
    return {
        "id": id,
        "type": type,
        "path": path,
        "date": date,
        "participants": [],
        "accounts": ["w-acct1"],
        "token_estimate": 10,
        "is_noise": is_noise,
        "is_distractor": False,
        "distractor_id": None,
        "model": model,
        "facts": facts or [],
    }


def _fact_placement(
    fact_id: str, locator: str, line_or_comment: str, excerpt: str
) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "locator": locator,
        "line_or_comment": line_or_comment,
        "excerpt": excerpt,
    }


def _truth_fact(fact_id: str, fact_type: str, claim: str, entities: list[str]) -> dict[str, Any]:
    return {
        "id": fact_id,
        "type": fact_type,
        "claim": claim,
        "entities": entities,
        "date": "2026-01-05",
    }


def _write_batch(
    batch_dir: Path, sources: list[dict[str, Any]], facts: list[dict[str, Any]]
) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator_version": "0.1.0",
        "schema_version": "0.1",
        "mode": "openrouter-free",
        "reduced": True,
        "generated_at": "2026-09-21T00:00:00+00:00",
        "truth_table_hash": "deadbeef",
        "sources": sources,
        "facts": {},
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    truth = {"meta": {"supplier": {"name": "Testvendor", "short": "Testvendor"}}, "facts": facts}
    with (batch_dir / "truth-table.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(truth, fh)


@pytest.fixture
def batch_dir(tmp_path: Path) -> Path:
    return tmp_path / "batch-fixture-01"


def _export(batch_dir: Path) -> list[dict[str, Any]]:
    """`training_export.export_batch(batch_dir)` with an EXPLICIT
    `out_path` under the batch's own `tmp_path` parent -- the library's
    own default (`training_export.default_export_path`) is a
    PROJECT-ROOT path, not test-isolated on its own; see this file's
    module docstring for the incident that made this the rule."""
    return training_export.export_batch(batch_dir, batch_dir.parent / "export.jsonl")


def test_export_call_source_with_verified_fact(batch_dir: Path):
    text = "[00:00:00] Speaker: The pump arrived late.\n[00:00:05] Speaker: Thanks, bye.\n"
    (batch_dir / "calls").mkdir(parents=True)
    (batch_dir / "calls" / "S-call-001.txt").write_text(text, encoding="utf-8")
    sources = [
        _source(
            id="S-call-001",
            type="call",
            path="calls/S-call-001.txt",
            facts=[_fact_placement("F1001", "00:00:00", "1", "the pump arrived late")],
        )
    ]
    facts = [_truth_fact("F1001", "complaint", "The pump arrived late.", ["w-acct1", "product-t2"])]
    _write_batch(batch_dir, sources, facts)

    records = _export(batch_dir)
    assert len(records) == 1
    r = records[0]
    assert r["record_id"] == "batch-fixture-01:S-call-001"
    assert r["corpus"] == "synthetic"
    assert r["kind"] == "call"
    assert r["license"] == "synthetic-owned"
    assert r["text"] == text
    assert r["teachers"] == ["faketeacher/alpha:free"]
    assert r["agreement"] == 1.0
    assert r["created"] == "2026-09-21T00:00:00+00:00"
    assert len(r["facts"]) == 1
    fact = r["facts"][0]
    assert fact["fact_type"] == "complaint"
    assert fact["claim"] == "The pump arrived late."
    assert fact["quote"] == "The pump arrived late."
    assert fact["quote"] in r["text"]
    assert fact["locator"] == "00:00:00"
    assert fact["entities"] == ["w-acct1", "product-t2"]


def test_export_noise_call_is_a_true_negative(batch_dir: Path):
    text = "[00:00:00] Speaker: Nothing to report.\n"
    (batch_dir / "calls").mkdir(parents=True)
    (batch_dir / "calls" / "S-call-n01.txt").write_text(text, encoding="utf-8")
    sources = [
        _source(
            id="S-call-n01", type="call", path="calls/S-call-n01.txt", is_noise=True, model=None
        )
    ]
    _write_batch(batch_dir, sources, facts=[])

    records = _export(batch_dir)
    assert len(records) == 1
    assert records[0]["facts"] == []
    assert records[0]["teachers"] == []


def test_export_skips_fact_bearing_source_that_failed_verification(batch_dir: Path):
    # is_noise=False (a fact WAS intended) but facts=[] (verification
    # failed both attempts, fell back to filler) -- must be skipped, never
    # exported as a false true-negative.
    text = "[00:00:00] Speaker: filler filler filler.\n"
    (batch_dir / "calls").mkdir(parents=True)
    (batch_dir / "calls" / "S-call-002.txt").write_text(text, encoding="utf-8")
    sources = [
        _source(
            id="S-call-002",
            type="call",
            path="calls/S-call-002.txt",
            is_noise=False,
            facts=[],
            model=None,
        )
    ]
    facts = [_truth_fact("F1002", "risk", "Some risk claim.", ["w-acct1"])]
    _write_batch(batch_dir, sources, facts)

    records = _export(batch_dir)
    assert records == []


def test_export_ticket_quote_is_comment_body(batch_dir: Path):
    (batch_dir / "tickets").mkdir(parents=True)
    payload = {
        "id": "S-ticket-001",
        "requester": "Jane Doe",
        "account": "Test Account",
        "created": "2026-01-05",
        "subject": "A question",
        "comments": [
            {"number": 1, "author": "Jane Doe", "body": "We were quoted a 10 percent discount."},
            {"number": 2, "author": "Support", "body": "Thanks, we will follow up."},
        ],
    }
    (batch_dir / "tickets" / "S-ticket-001.json").write_text(json.dumps(payload), encoding="utf-8")
    sources = [
        _source(
            id="S-ticket-001",
            type="ticket",
            path="tickets/S-ticket-001.json",
            facts=[_fact_placement("F1003", "S-ticket-001#1", "1", "quoted a 10 percent discount")],
        )
    ]
    facts = [
        _truth_fact(
            "F1003", "pricing", "Asked for a 10 percent discount.", ["w-acct1", "product-t2"]
        )
    ]
    _write_batch(batch_dir, sources, facts)

    records = _export(batch_dir)
    assert len(records) == 1
    fact = records[0]["facts"][0]
    assert fact["quote"] == "We were quoted a 10 percent discount."
    assert fact["quote"] in records[0]["text"]
    assert "[1] Jane Doe: We were quoted a 10 percent discount." in records[0]["text"]


def test_export_email_quote_is_paragraph(batch_dir: Path):
    (batch_dir / "email").mkdir(parents=True)
    file_text = (
        "Message-ID: <S-email-001@halvardmedical.com>\n"
        "From: Jane Doe <jane@example.org>\n"
        "To: Support <support@example.com>\n"
        "Date: 2026-01-05 09:00:00 -0700\n"
        "Subject: Following up\n\n"
        "Hi there,\n\n"
        "We are evaluating Corvane pumps as an alternative.\n\n"
        "Best,\nJane\n"
    )
    (batch_dir / "email" / "S-email-001.txt").write_text(file_text, encoding="utf-8")
    sources = [
        _source(
            id="S-email-001",
            type="email",
            path="email/S-email-001.txt",
            facts=[
                _fact_placement(
                    "F1004", "S-email-001@halvardmedical.com#p2", "2", "evaluating Corvane"
                )
            ],
        )
    ]
    facts = [
        _truth_fact(
            "F1004", "competitor_mention", "Evaluating a competitor.", ["w-acct1", "corvane"]
        )
    ]
    _write_batch(batch_dir, sources, facts)

    records = _export(batch_dir)
    fact = records[0]["facts"][0]
    assert fact["quote"] == "We are evaluating Corvane pumps as an alternative."
    assert fact["quote"] in records[0]["text"]


def test_export_groups_chat_messages_by_file_and_uses_bare_message_text(batch_dir: Path):
    (batch_dir / "slack" / "acct-x").mkdir(parents=True)
    messages = [
        {
            "type": "message",
            "user": "w-jane",
            "text": "hey, quick heads up",
            "ts": "1.0",
            "thread_ts": None,
        },
        {
            "type": "message",
            "user": "w-jane",
            "text": "we are looking at a 12 percent discount ask",
            "ts": "2.0",
            "thread_ts": None,
        },
    ]
    (batch_dir / "slack" / "acct-x" / "2026-01-05.json").write_text(
        json.dumps(messages), encoding="utf-8"
    )
    users = [{"id": "w-jane", "name": "jane", "real_name": "Jane Doe", "is_bot": False}]
    (batch_dir / "slack").mkdir(exist_ok=True)
    (batch_dir / "slack" / "users.json").write_text(json.dumps(users), encoding="utf-8")

    sources = [
        _source(
            id="S-slack-001",
            type="slack",
            path="slack/acct-x/2026-01-05.json",
            is_noise=True,
            model=None,
        ),
        _source(
            id="S-slack-002",
            type="slack",
            path="slack/acct-x/2026-01-05.json",
            facts=[
                _fact_placement(
                    "F1005", "slack/acct-x/2026-01-05.json#m2", "2", "12 percent discount"
                )
            ],
        ),
    ]
    facts = [_truth_fact("F1005", "pricing", "Asked for a 12 percent discount.", ["w-acct1"])]
    _write_batch(batch_dir, sources, facts)

    records = _export(batch_dir)
    assert len(records) == 1
    r = records[0]
    assert r["kind"] == "chat"
    assert r["source_id"] == "slack/acct-x/2026-01-05.json"
    assert (
        r["text"]
        == "Jane Doe: hey, quick heads up\nJane Doe: we are looking at a 12 percent discount ask"
    )
    fact = r["facts"][0]
    assert fact["quote"] == "we are looking at a 12 percent discount ask"
    assert fact["quote"] in r["text"]
    assert "Jane Doe:" not in fact["quote"]


def test_export_crm_sources_never_exported(batch_dir: Path):
    (batch_dir / "crm" / "activity_notes").mkdir(parents=True)
    sources = [
        _source(
            id="S-crm-note-N-8001",
            type="crm",
            path="crm/activity_notes/N-8001.json",
            facts=[
                _fact_placement("F1006", "crm/activity_notes/N-8001.json#p0", "0", "usage claim")
            ],
        )
    ]
    facts = [_truth_fact("F1006", "usage", "Runs 10 pumps.", ["w-acct1"])]
    _write_batch(batch_dir, sources, facts)

    records = _export(batch_dir)
    assert records == []


def test_export_writes_jsonl_and_default_path(batch_dir: Path, tmp_path: Path):
    text = "[00:00:00] Speaker: The pump arrived late.\n"
    (batch_dir / "calls").mkdir(parents=True)
    (batch_dir / "calls" / "S-call-001.txt").write_text(text, encoding="utf-8")
    sources = [
        _source(
            id="S-call-001",
            type="call",
            path="calls/S-call-001.txt",
            facts=[_fact_placement("F1001", "00:00:00", "1", "the pump arrived late")],
        )
    ]
    facts = [_truth_fact("F1001", "complaint", "The pump arrived late.", ["w-acct1"])]
    _write_batch(batch_dir, sources, facts)

    out_path = tmp_path / "custom" / "out.jsonl"
    records = training_export.export_batch(batch_dir, out_path)
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records) == 1
    assert json.loads(lines[0])["record_id"] == records[0]["record_id"]

    default_path = training_export.default_export_path(batch_dir)
    assert default_path.name == "batch-fixture-01.jsonl"
    assert default_path.parent == training_export.DEFAULT_SILVER_DIR


def test_export_record_ids_are_unique(batch_dir: Path):
    (batch_dir / "calls").mkdir(parents=True)
    (batch_dir / "calls" / "S-call-001.txt").write_text(
        "[00:00:00] Speaker: x.\n", encoding="utf-8"
    )
    (batch_dir / "calls" / "S-call-002.txt").write_text(
        "[00:00:00] Speaker: y.\n", encoding="utf-8"
    )
    sources = [
        _source(
            id="S-call-001", type="call", path="calls/S-call-001.txt", is_noise=True, model=None
        ),
        _source(
            id="S-call-002", type="call", path="calls/S-call-002.txt", is_noise=True, model=None
        ),
    ]
    _write_batch(batch_dir, sources, facts=[])
    records = _export(batch_dir)
    ids = [r["record_id"] for r in records]
    assert len(ids) == len(set(ids)) == 2


# --------------------------------------------------------------------------
# End-to-end smoke: real cli.synth() orchestration + a fake, no-network
# teacher (tests/synth_fakes.py).
# --------------------------------------------------------------------------


def test_end_to_end_synth_then_export(tmp_path: Path):
    out_dir = tmp_path / "batch-e2e-01"
    report = run_fake_synth(seed=11, out_dir=out_dir, num_accounts=2, per_kind=6)
    assert report["manifest_problems"] == []
    assert report["verified"] > 0

    # Explicit out_path under tmp_path -- never the project-root default
    # (training_export.default_export_path) from a test. See this file's
    # module docstring.
    export_path = tmp_path / "export.jsonl"
    records = training_export.export_batch(out_dir, export_path)
    assert records
    kinds = {r["kind"] for r in records}
    assert kinds <= {"call", "ticket", "email", "chat"}
    assert "crm" not in kinds

    for r in records:
        for fact in r["facts"]:
            assert fact["quote"] in r["text"]
        assert r["agreement"] == 1.0
        assert r["corpus"] == "synthetic"
        assert r["license"] == "synthetic-owned"

    ids = [r["record_id"] for r in records]
    assert len(ids) == len(set(ids))

    assert export_path.exists()
    lines = export_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    assert not training_export.default_export_path(out_dir).exists()
