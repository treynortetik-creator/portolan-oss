"""Tests for `portolan.training.pipeline`'s document readers --
`documents_for_corpus` (the `FileConnector`-backed corpora) and
`_iter_ubuntu_irc_documents` (the bespoke chat windower) -- plus the
hold-out guard wired into both (silver-labels build instructions item 5).
Every fixture here is built fresh under `tmp_path`, never
`corpus/out/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from portolan.extract.schema import SourceType
from portolan.training.pipeline import (
    CORPUS_KINDS,
    CORPUS_LICENSES,
    _iter_ubuntu_irc_documents,
    documents_for_corpus,
)

# --------------------------------------------------------------------------
# FileConnector-backed corpora (fda_maude/ami/enron)
# --------------------------------------------------------------------------


def test_documents_for_corpus_fda_maude_tags_kind_ticket(tmp_path: Path) -> None:
    root = tmp_path / "fda_maude"
    (root / "tickets").mkdir(parents=True)
    ticket = {
        "id": "S-ticket-fda-1",
        "requester": "Reporter",
        "account": "Acme Devices",
        "created": "2026-01-05",
        "subject": "Pump malfunction",
        "comments": [{"number": 1, "author": "Reporter", "body": "the pump stopped working"}],
    }
    (root / "tickets" / "S-ticket-fda-1.json").write_text(json.dumps(ticket), encoding="utf-8")

    docs = list(documents_for_corpus("fda_maude", root))
    assert len(docs) == 1
    assert docs[0].source_id == "S-ticket-fda-1"
    assert docs[0].kind == "ticket"
    assert docs[0].raw.source_type == SourceType.TICKET
    assert "pump stopped working" in docs[0].raw.text


def test_documents_for_corpus_ami_tags_kind_call(tmp_path: Path) -> None:
    root = tmp_path / "ami"
    (root / "calls").mkdir(parents=True)
    (root / "calls" / "EN2001a.txt").write_text(
        "[00:00:01] Speaker A: let's talk about the remote control design\n",
        encoding="utf-8",
    )

    docs = list(documents_for_corpus("ami", root))
    assert len(docs) == 1
    assert docs[0].kind == "call"
    assert docs[0].raw.source_type == SourceType.CALL


def test_documents_for_corpus_enron_tags_kind_email(tmp_path: Path) -> None:
    root = tmp_path / "enron"
    (root / "email").mkdir(parents=True)
    (root / "email" / "msg-001.txt").write_text(
        "Message-ID: <msg-001>\nFrom: alice@enron.com\nTo: bob@enron.com\n"
        "Date: 2026-01-05\nSubject: deal terms\n\nLet's finalize the terms.\n",
        encoding="utf-8",
    )

    docs = list(documents_for_corpus("enron", root))
    assert len(docs) == 1
    assert docs[0].kind == "email"
    assert docs[0].raw.source_type == SourceType.EMAIL


def test_documents_for_corpus_cfpb_tags_kind_ticket(tmp_path: Path) -> None:
    root = tmp_path / "cfpb"
    (root / "tickets").mkdir(parents=True)
    ticket = {
        "id": "S-ticket-cfpb-1",
        "requester": "consumer",
        "account": "Big Bank Inc.",
        "created": "2020-01-05",
        "subject": "Debt collection",
        "comments": [{"number": 1, "author": "Consumer", "body": "they keep calling me"}],
    }
    (root / "tickets" / "S-ticket-cfpb-1.json").write_text(json.dumps(ticket), encoding="utf-8")

    docs = list(documents_for_corpus("cfpb", root))
    assert len(docs) == 1
    assert docs[0].source_id == "S-ticket-cfpb-1"
    assert docs[0].kind == "ticket"
    assert docs[0].raw.source_type == SourceType.TICKET
    assert "they keep calling me" in docs[0].raw.text


def test_documents_for_corpus_saas_sales_tags_kind_call(tmp_path: Path) -> None:
    root = tmp_path / "saas_sales"
    (root / "calls").mkdir(parents=True)
    (root / "calls" / "S-call-saas-1-conv-0.txt").write_text(
        "[00:00:00] Customer: hi, what's the pricing?\n\n[00:00:30] Sales Rep: it's $99/month\n",
        encoding="utf-8",
    )

    docs = list(documents_for_corpus("saas_sales", root))
    assert len(docs) == 1
    assert docs[0].kind == "call"
    assert docs[0].raw.source_type == SourceType.CALL


def test_corpus_registry_includes_cfpb_and_saas_sales() -> None:
    assert CORPUS_KINDS["cfpb"] == "ticket"
    assert CORPUS_KINDS["saas_sales"] == "call"
    assert "cfpb" in CORPUS_LICENSES
    assert "saas_sales" in CORPUS_LICENSES
    assert "apache" in CORPUS_LICENSES["saas_sales"].lower()
    assert "public domain" in CORPUS_LICENSES["cfpb"].lower()


# --------------------------------------------------------------------------
# ubuntu_irc windowing
# --------------------------------------------------------------------------


def _write_ubuntu_fixture(tmp_path: Path, messages: list[dict]) -> tuple[Path, Path]:
    slack_export = tmp_path / "ubuntu_irc" / "slack_export"
    channel_dir = slack_export / "ubuntu"
    channel_dir.mkdir(parents=True)
    (channel_dir / "2026-01-05.json").write_text(json.dumps(messages), encoding="utf-8")
    users = [
        {"id": "U1", "profile": {"real_name": "alice"}},
        {"id": "U2", "profile": {"real_name": "bob"}},
    ]
    users_path = slack_export / "users.json"
    users_path.write_text(json.dumps(users), encoding="utf-8")
    return channel_dir, users_path


def test_ubuntu_irc_windowing_splits_into_chunks(tmp_path: Path) -> None:
    messages = [
        {"type": "message", "user": "U1", "text": f"message number {i}", "ts": str(i)}
        for i in range(5)
    ]
    channel_dir, users_path = _write_ubuntu_fixture(tmp_path, messages)

    docs = list(_iter_ubuntu_irc_documents(channel_dir, users_path, window_size=2))
    assert len(docs) == 3  # 5 messages / window 2 -> 2, 2, 1
    assert [d.source_id for d in docs] == [
        "ubuntu-2026-01-05-w001",
        "ubuntu-2026-01-05-w002",
        "ubuntu-2026-01-05-w003",
    ]
    assert docs[0].kind == "chat"
    assert docs[0].raw.source_type == SourceType.CHAT
    assert docs[0].raw.text == "alice: message number 0\nalice: message number 1"
    assert docs[2].raw.text == "alice: message number 4"


def test_ubuntu_irc_resolves_usernames_and_skips_blank_messages(tmp_path: Path) -> None:
    messages = [
        {"type": "message", "user": "U1", "text": "hello there", "ts": "1"},
        {"type": "message", "user": "U2", "text": "", "ts": "2"},  # blank -- skipped
        {"type": "message", "user": "U2", "text": "how do I fix this", "ts": "3"},
    ]
    channel_dir, users_path = _write_ubuntu_fixture(tmp_path, messages)

    docs = list(_iter_ubuntu_irc_documents(channel_dir, users_path, window_size=10))
    assert len(docs) == 1
    assert docs[0].raw.text == "alice: hello there\nbob: how do I fix this"
    names = {p.name for p in docs[0].raw.participants}
    assert names == {"alice", "bob"}


def test_ubuntu_irc_skips_non_message_events(tmp_path: Path) -> None:
    messages = [
        {"type": "channel_join", "user": "U1", "text": "alice has joined", "ts": "1"},
        {"type": "message", "user": "U1", "text": "hi", "ts": "2"},
    ]
    channel_dir, users_path = _write_ubuntu_fixture(tmp_path, messages)

    docs = list(_iter_ubuntu_irc_documents(channel_dir, users_path, window_size=10))
    assert len(docs) == 1
    assert docs[0].raw.text == "alice: hi"


# --------------------------------------------------------------------------
# Hold-out guard, wired into the real readers
# --------------------------------------------------------------------------


def test_documents_for_corpus_refuses_corpus_out(tmp_path: Path) -> None:
    forbidden = tmp_path / "corpus" / "out"
    (forbidden / "tickets").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="hold-out violation"):
        list(documents_for_corpus("fda_maude", forbidden))


def test_ubuntu_irc_reader_refuses_corpus_out(tmp_path: Path) -> None:
    channel_dir = tmp_path / "corpus" / "out" / "slack_export" / "ubuntu"
    channel_dir.mkdir(parents=True)
    users_path = channel_dir.parent / "users.json"
    users_path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hold-out violation"):
        list(_iter_ubuntu_irc_documents(channel_dir, users_path))
