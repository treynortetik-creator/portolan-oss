"""The Monday email (`portolan.report.monday`, `01-spec/monday-email-2026-09-25.md`)
on the demo graph: the synthetic corpus through the pipeline, then the demo's
weekly CRM exports (`corpus/demo/weekly-crm/`) as snapshots. Every link in the
email must open the cited original at the cited block, with the quoted span on
it, the way `scripts/verify.py` checks a chat citation."""

from __future__ import annotations

import importlib.util
import re
import smtplib
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from portolan.config import Settings
from portolan.connectors.file_connector import FileConnector
from portolan.crm.ingest import ingest_crm_snapshots
from portolan.fake_extractor import FakePlacements, build_fake_llm_client
from portolan.llm.fake import FakeLLMClient
from portolan.pipeline import run
from portolan.registry import Registry
from portolan.report import monday
from portolan.report.monday import AccountBlock, Cite, Item, Sentence
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.web.app import create_app
from portolan.web.deps import get_conn, get_store_dep
from tests._paths import require_repo_path

CORPUS_OUT = Path(__file__).resolve().parent.parent / "corpus" / "out"
WEEKLY_CRM = CORPUS_OUT.parent / "demo" / "weekly-crm"
ORG_KEY = "test-org-key-for-the-monday-email"


def _load_verify():
    path = require_repo_path("scripts", "verify.py")
    spec = importlib.util.spec_from_file_location("verify_monday_email", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_monday_email"] = module
    spec.loader.exec_module(module)
    return module


verify = _load_verify()


@pytest.fixture()
def demo(db, tmp_path: Path):
    """The demo graph as `python -m portolan.web.demo` builds it."""
    settings = Settings(
        database_url="postgresql://localhost/unused",
        tenant="monday-test",
        store_root=tmp_path / "s",
        crm_change_claims=True,
    )
    store = LocalStore(settings.store_root)
    report = run(
        db,
        connector=FileConnector(CORPUS_OUT),
        llm=build_fake_llm_client(FakePlacements(CORPUS_OUT), tenant=settings.tenant),
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=store,
        settings=settings,
    )
    weekly = ingest_crm_snapshots(db, FileConnector(WEEKLY_CRM), store, embedder=FakeEmbedder())
    return report, weekly, store


def _client(db, db_url: str, store: LocalStore, tmp_path: Path) -> TestClient:
    settings = Settings(
        database_url=db_url,
        tenant="test",
        store_root=tmp_path / "s",
        llm_provider="fake",
        embed_provider="fake",
        org_key=ORG_KEY,
        session_secret="test-session-secret-for-the-monday-email-only",
        cookie_secure=False,
    )
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_store_dep] = lambda: store
    return TestClient(app)


def test_the_demo_email_ranks_changes_checks_the_crm_and_every_link_opens_its_source(
    demo, db, db_url: str, tmp_path: Path
):
    run_report, weekly, store = demo
    # The pipeline itself turns the corpus's two opportunity exports into one change.
    assert run_report.crm_changes_written == 1
    assert weekly.changes.changes_written == 28

    email = monday.build_monday(db, week="2026-W35", owner="Sam Okafor")

    assert email.owner_crm_id == "sam-okafor"
    assert [b.name for b in email.accounts] == [
        "Northlake Health System",
        "Ashgrove Health",
        "Meadowbrook Community Hospital",
        "Cobalt Ridge Medical Center",
        "Larkspur Regional Health",
        "Larkspur North Hospital",
    ]
    assert [b.score for b in email.accounts] == [24, 23, 20, 17, 14, 3]
    assert "Anchor Bay Surgical Partners" in [b.name for b in email.quiet]
    assert email.counts() == {"truth": 3, "crm": 23, "call": 1, "action": 2}
    assert email.unwritten_crm_changes == 0

    ashgrove = email.accounts[1]
    truth = [i for i in ashgrove.items if i.kind == "truth"]
    assert len(truth) == 1
    assert truth[0].sentences[0].text == (
        "The CRM moved forward this week: Ashgrove Health — HalvardCare renewal "
        "(forecast category, health, stage)."
    )
    assert (
        truth[0]
        .sentences[1]
        .text.startswith("But a call on 2026-08-20 said: Ashgrove Health's CFO said on 2026-08-20")
    )
    assert truth[0].sentences[1].cites[0].source_slug == "S-call-013"
    overdue = [i for i in ashgrove.items if i.kind == "action"]
    assert overdue[0].sentences[0].text.startswith("Overdue (due 2026-08-28): Sam Okafor promised")
    assert "[[" not in monday.render_text(email)

    cites = [
        c
        for block in email.accounts
        for item in block.items
        for sentence in item.sentences
        for c in sentence.cites
    ]
    assert all(s.cites for b in email.accounts for i in b.items for s in i.sentences)
    unique = {(c.source_slug, c.locator): c for c in cites}
    assert len(unique) == 32

    with _client(db, db_url, store, tmp_path) as client:
        login = client.post(
            "/login", data={"org_key": ORG_KEY, "editor": "Test"}, follow_redirects=False
        )
        assert login.status_code == 303
        failures = []
        for cite in unique.values():
            href = cite.href("")
            response = client.get(href)
            citation = {"original_href": href, "quote": cite.quote}
            problems: list[str] = []
            if response.status_code != 200:
                problems.append(f"{response.status_code}")
            else:
                verify._check_original(citation, verify.Page(response), problems)
                if citation.get("cited_block_found") is not True:
                    problems.append("no cited block")
            if problems:
                failures.append((href, problems))
    assert failures == []
    # Every CRM link carried a quote that was found on its row. The call claims and actions
    # below carry none in the synthetic corpus; their cited block was still found.
    assert {(c.source_slug, c.locator) for c in unique.values() if not c.quote} == {
        ("S-call-013", "09:27:31"),
        ("S-call-013", "09:25:24"),
        ("S-call-014", "09:12:50"),
        ("S-call-018", "09:23:30"),
    }


def test_the_html_uses_the_brand_tokens_and_every_href_is_a_listed_source(demo, db):
    email = monday.build_monday(db, week="2026-W35", owner="sam-okafor", base_url="http://x")
    page = monday.render_html(email)
    tokens = monday.brand_tokens()

    assert tokens["ground"] == "#FAF8F4" and tokens["accent"] == "#C0362C"
    assert f"background:{tokens['ground']}" in page
    assert "var(--pl-" not in page
    hrefs = set(re.findall(r'href="([^"]+)"', page))
    listed = set(monday.to_json(email)["links"])
    assert {h.replace("&amp;", "&") for h in hrefs} == listed
    assert all(urlsplit(h).path.startswith("/originals/") for h in listed)
    assert "No changes this week: " in page
    text = monday.render_text(email)
    assert text.count("\n[") == len(listed)


def test_top_n_takes_the_largest_accounts_by_open_pipeline_then_arr(demo, db):
    email = monday.build_monday(db, week="2026-W35", owner="sam-okafor", top_n=3)
    names = {b.name for b in [*email.accounts, *email.quiet]}
    # Open pipeline: Ashgrove 2,350,000, Larkspur Regional 1,562,000; then ARR 1,600,000.
    assert names == {"Ashgrove Health", "Ashgrove Medical Center", "Larkspur Regional Health"}

    chosen = monday.build_monday(
        db, week="2026-W35", account_ids=["A-1040", "Anchor Bay Surgical Partners"]
    )
    assert [b.name for b in chosen.accounts] == ["Meadowbrook Community Hospital"]
    assert [b.name for b in chosen.quiet] == ["Anchor Bay Surgical Partners"]


def test_the_cli_writes_three_files_and_send_refuses_without_an_smtp_setting(
    demo, db_url: str, tmp_path: Path, monkeypatch
):
    from portolan.report.weekly import main

    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.delenv("PORTOLAN_SMTP_URL", raising=False)
    out = tmp_path / "out"
    code = main(
        ["monday", "--owner", "sam-okafor", "--week", "2026-W35", "--out", str(out), "--send",
         "--to", "rep@example.com"]
    )  # fmt: skip

    assert code == 3
    assert sorted(p.name for p in out.iterdir()) == [
        "monday-sam-okafor-2026-W35.html",
        "monday-sam-okafor-2026-W35.json",
        "monday-sam-okafor-2026-W35.txt",
    ]


def test_send_goes_through_the_configured_server_only(monkeypatch):
    with pytest.raises(monday.SendRefused):
        monday.send_email(
            Settings(smtp_url=None), to="a@b.c", subject="s", html_body="h", text_body="t"
        )

    sent = []

    class FakeSMTP:
        def __init__(self, host, port):
            sent.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            sent.append(("starttls",))

        def login(self, user, password):
            sent.append(("login", user))

        def send_message(self, message):
            sent.append(("send", message["To"], message.get_content_type()))

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    settings = Settings(smtp_url="smtp://mailer:pw@mail.example.com:2525", smtp_from="p@x.io")
    monday.send_email(
        settings, to="rep@example.com", subject="s", html_body="<p>h</p>", text_body="t"
    )
    assert sent == [
        ("connect", "mail.example.com", 2525),
        ("starttls",),
        ("login", "mailer"),
        ("send", "rep@example.com", "multipart/alternative"),
    ]


def _block() -> AccountBlock:
    block = AccountBlock("A-1", "Acme Health", None, amount=Decimal(0), amount_basis="")
    block.items = [
        Item("crm", "forward", [Sentence(
            "Acme renewal: amount rose from $100,000 to $125,000, up $25,000 or 25.0% "
            "(CRM, 2026-08-21).",
            [Cite("S-crm-2", "crm/o.csv#O-1", "CRM"), Cite("S-crm-1", "crm/p.csv#O-1", "was")],
        )]),
        Item("call", "risk", [Sentence(
            "The CFO said late deliveries would come up at the 2027 renewal.",
            [Cite("S-call-9", "00:04:10", "call")],
        )]),
    ]  # fmt: skip
    return block


def test_a_summary_keeps_only_sentences_that_cite_given_lines_and_invent_no_number():
    good = (
        "The renewal grew by $25,000 ^[S-crm-2 @ crm/o.csv#O-1] as the CFO warned about "
        "deliveries ^[S-call-9 @ 00:04:10]."
    )
    uncited = "Momentum is strong this quarter."
    invented = "The deal is now worth $150,000 ^[S-crm-2 @ crm/o.csv#O-1]."
    wrong_source = "The CFO is unhappy ^[S-call-1 @ 00:00:01]."
    borrowed = "The call mentioned $25,000 ^[S-call-9 @ 00:04:10]."
    llm = FakeLLMClient([{"sentences": [good, uncited]}, {"sentences": [invented, wrong_source]},
                         {"sentences": [borrowed]}])  # fmt: skip
    notes: list[str] = []

    assert monday._summarize(llm, _block(), prose_model=None, notes=notes) == [good]
    assert monday._summarize(llm, _block(), prose_model=None, notes=notes) == []
    # A number licensed only under the CRM row's citation cannot move to the call's.
    assert monday._summarize(llm, _block(), prose_model=None, notes=notes) == []
    assert notes == [
        "Acme Health: 1 summary sentence(s) dropped by the citation checks",
        "Acme Health: 2 summary sentence(s) dropped by the citation checks",
        "Acme Health: 1 summary sentence(s) dropped by the citation checks",
    ]
    block = _block()
    block.summary = [good]
    email = monday.MondayEmail(
        owner_crm_id=None, owner_name="", week="2026-W35", start=date(2026, 8, 24),
        end=date(2026, 8, 30), base_url="", accounts=[block], quiet=[],
    )  # fmt: skip
    page = monday.render_html(email)
    assert "^[" not in page
    assert page.count('href="/originals/S-crm-2?at=crm%2Fo.csv%23O-1"') == 2
    assert "^[" not in monday.render_text(email)
