"""`scripts/verify.py` (guardrails plan 2026-09-22, rank 8) against a real
app: `create_app` under uvicorn on 127.0.0.1, on the session's test
database seeded by `tests/verify_seed.py`, driven over real HTTP by the
script's own `main()`.

The point of the script is that a broken page FAILS, so most tests here
break something on purpose and check the result file says so: an Actions
page that renders but is empty, a Review page that 500s, a wrong MCP key.
Breaking is done by patching the route module's own import in this process
(the server thread shares it), and undone by `monkeypatch` after the test.
"""

from __future__ import annotations

import importlib.util
import json
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest

from portolan.mcp_server.server import TOOL_NAMES
from tests._paths import require_repo_path
from tests.verify_seed import app_settings, seed_verify_graph, serve, write_keys

REPO_ROOT = Path(__file__).resolve().parent.parent
# The image leaves scripts/ out; this skips the module at collection there.
_VERIFY = require_repo_path("scripts", "verify.py")


def _load_verify():
    spec = importlib.util.spec_from_file_location("verify", _VERIFY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify"] = module
    spec.loader.exec_module(module)
    return module


verify = _load_verify()


@dataclass
class VerifyApp:
    base_url: str
    org_key: str
    mcp_key: str
    org_key_file: Path
    mcp_key_file: Path
    counts: dict[str, int]


@pytest.fixture(scope="module")
def verify_app(db_url: str, tmp_path_factory: pytest.TempPathFactory):
    """The session's test database, rebuilt and seeded once for this
    module -- the same shape as `tests/test_web.py`'s `demo_conn` (a
    second database would add a CREATE and a DROP DATABASE, about 40 s
    each on a busy dev Postgres)."""
    try:
        psycopg.connect(db_url).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unreachable at {db_url!r}: {exc}")
    workdir = tmp_path_factory.mktemp("verify-app")
    store_root = workdir / "store"
    counts = seed_verify_graph(db_url, store_root)
    org_key, mcp_key, org_path, mcp_path = write_keys(workdir / "keys")
    settings = app_settings(
        db_url,
        store_root,
        org_key=org_key,
        session_secret=secrets.token_hex(32),
        mcp_org_key=mcp_key,
    )
    with serve(settings) as base_url:
        yield VerifyApp(base_url, org_key, mcp_key, org_path, mcp_path, counts)


def _run(app: VerifyApp, target: str, out_dir: Path, *extra: str) -> tuple[int, dict, Path]:
    code = verify.main(
        [
            target,
            "--base-url",
            app.base_url,
            "--org-key-file",
            str(app.org_key_file),
            "--mcp-key-file",
            str(app.mcp_key_file),
            "--out-dir",
            str(out_dir),
            *extra,
        ]
    )
    files = sorted(out_dir.glob(f"*-{target}.json"))
    assert len(files) == 1, files
    return code, json.loads(files[0].read_text(encoding="utf-8")), files[0]


def _failed(result: dict) -> list[dict]:
    return [c for c in result["checks"] if not c["ok"]]


def test_seed_did_the_work(verify_app: VerifyApp) -> None:
    counts = verify_app.counts
    assert counts["notes"] > 0
    assert counts["open_actions_with_owner_and_due"] >= 1
    assert counts["open_review_rows"] >= 1
    assert counts["reports"] == 1
    assert counts["coverage_rows"] >= 1


def test_all_passes_against_a_working_app(verify_app: VerifyApp, tmp_path: Path, capsys) -> None:
    code, result, path = _run(verify_app, "all", tmp_path)

    assert code == 0, _failed(result)
    assert result["ok"] is True
    assert result["passed"] == result["total"]
    assert {c["target"] for c in result["checks"]} == set(verify.TARGETS)
    for check in result["checks"]:
        if check["status"] is not None:
            assert check["elapsed_ms"] >= 0 and check["bytes"] >= 0
        for marker in check.get("markers", []):
            assert marker["ok"], (check["name"], marker)
    # Markers counted real data, not just a 200.
    by_name = {c["name"]: c for c in result["checks"]}
    actions = {m["name"]: m["found"] for m in by_name["open actions by owner"]["markers"]}
    assert actions["action rows with an owner and a due date"] >= 1
    report = {m["name"]: m["found"] for m in by_name["newest weekly report"]["markers"]}
    assert report["action cards under Suggested actions"] >= 1
    assert result["derived"]["report_slug"] == "report-2026-W36"
    assert set(verify.REQUIRED_MCP_TOOLS) <= set(result["derived"]["mcp_tools"])

    line = capsys.readouterr().out.strip()
    assert line.startswith(f"PASS {result['total']}/{result['total']} -> ")
    assert line.endswith(path.name)


def test_result_file_and_stdout_never_carry_a_key(
    verify_app: VerifyApp, tmp_path: Path, capsys
) -> None:
    _code, _result, path = _run(verify_app, "all", tmp_path)
    written = path.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    for secret in (verify_app.org_key, verify_app.mcp_key):
        assert secret not in written
        assert secret not in out
    assert "file:" in json.loads(written)["auth"]["org_key"]


def test_an_empty_actions_page_fails_on_its_marker_not_its_status(
    verify_app: VerifyApp, tmp_path: Path, monkeypatch, capsys
) -> None:
    """The page still renders 200 with its "No open actions" empty state --
    exactly the break a status check alone would miss."""
    monkeypatch.setattr("portolan.web.routes.actions.actions_feed", lambda conn: [])

    code, result, _path = _run(verify_app, "actions", tmp_path)

    assert code == 1
    (failed,) = _failed(result)
    assert failed["name"] == "open actions by owner"
    assert failed["status"] == 200
    bad = {m["name"]: m["found"] for m in failed["markers"] if not m["ok"]}
    assert bad == {
        "open count in the summary line": 0,
        "action rows with an owner and a due date": 0,
    }
    assert capsys.readouterr().out.startswith("FAIL ")


def test_a_review_page_that_errors_fails_on_status(
    verify_app: VerifyApp, tmp_path: Path, monkeypatch
) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("broken on purpose")

    monkeypatch.setattr("portolan.web.routes.review.review_queue_open_index", _boom)

    code, result, _path = _run(verify_app, "review", tmp_path)

    assert code == 1
    review = next(c for c in result["checks"] if c["name"] == "open review queue")
    assert review["ok"] is False
    assert review["status"] == 500


def test_a_report_without_action_cards_fails(
    verify_app: VerifyApp, tmp_path: Path, monkeypatch
) -> None:
    from markupsafe import Markup

    monkeypatch.setattr(
        "portolan.web.routes.reports.render_note_body",
        lambda conn, body: Markup("<h3>Themes by movement</h3><ul><li>no cards</li></ul>"),
    )

    code, result, _path = _run(verify_app, "reports", tmp_path)

    assert code == 1
    report = next(c for c in result["checks"] if c["name"] == "newest weekly report")
    bad = [m["name"] for m in report["markers"] if not m["ok"]]
    assert bad == ["action cards under Suggested actions"]


def test_a_wrong_mcp_key_fails_and_is_not_recorded(verify_app: VerifyApp, tmp_path: Path) -> None:
    wrong = tmp_path / "wrong-mcp-key.txt"
    wrong_value = "not-the-key-" + secrets.token_hex(16)
    wrong.write_text(wrong_value, encoding="utf-8")
    out_dir = tmp_path / "out"

    code = verify.main(
        ["mcp", "--base-url", verify_app.base_url, "--mcp-key-file", str(wrong)]
        + ["--out-dir", str(out_dir)]
    )

    assert code == 1
    (path,) = out_dir.glob("*-mcp.json")
    written = path.read_text(encoding="utf-8")
    assert wrong_value not in written
    result = json.loads(written)
    init = next(c for c in result["checks"] if c["name"] == "MCP initialize")
    assert init["status"] == 401
    # The unauthenticated probe still passes: the door is locked.
    probe = next(c for c in result["checks"] if c["name"] == "MCP refuses a request with no key")
    assert probe["ok"] is True


def test_a_missing_org_key_fails_login_without_a_request(
    verify_app: VerifyApp, tmp_path: Path
) -> None:
    code = verify.main(
        ["login", "--base-url", verify_app.base_url]
        + ["--org-key-file", str(tmp_path / "absent.txt"), "--out-dir", str(tmp_path)]
    )

    assert code == 1
    (path,) = tmp_path.glob("*-login.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    post = next(c for c in result["checks"] if c["name"] == "login POST")
    assert post["status"] is None
    assert "missing" in post["error"]


def test_the_scrubber_redacts_and_fails_a_result_that_holds_a_secret(tmp_path: Path) -> None:
    serialized, scrubbed = verify._scrub('{"x": "abc-SECRET-def"}', ["SECRET"])
    assert scrubbed is True
    assert "SECRET" not in serialized


def test_required_mcp_tools_are_real_tools() -> None:
    assert set(verify.REQUIRED_MCP_TOOLS) <= set(TOOL_NAMES)


def test_key_resolution_prefers_the_flag_then_the_file_then_the_env(
    tmp_path: Path, monkeypatch
) -> None:
    flag = tmp_path / "flag.txt"
    flag.write_text("  from-flag\n", encoding="utf-8")
    default = tmp_path / "default.txt"
    monkeypatch.setenv("VERIFY_TEST_KEY", "from-env")

    assert verify.resolve_key(flag, default, "VERIFY_TEST_KEY").value == "from-flag"
    env_key = verify.resolve_key(None, default, "VERIFY_TEST_KEY")
    assert (env_key.value, env_key.source) == ("from-env", "env:VERIFY_TEST_KEY")
    default.write_text("from-file", encoding="utf-8")
    assert verify.resolve_key(None, default, "VERIFY_TEST_KEY").value == "from-file"
    monkeypatch.delenv("VERIFY_TEST_KEY")
    assert verify.resolve_key(None, tmp_path / "none.txt", "VERIFY_TEST_KEY").value is None


def test_seed_opens_the_ai_access_door(verify_app: VerifyApp) -> None:
    # Migration 0031 leaves a new database closed; the seed opens it for its tenant.
    assert verify_app.counts["ai_access_open"] == 1


def test_verify_knows_the_servers_door_closed_code() -> None:
    from portolan.mcp_server.access import AI_ACCESS_CLOSED_CODE

    assert verify.AI_ACCESS_CLOSED_CODE == AI_ACCESS_CLOSED_CODE


def test_a_closed_ai_access_door_fails_mcp_with_a_hint(
    verify_app: VerifyApp, db_url: str, tmp_path: Path, capsys
) -> None:
    from portolan import db as db_module
    from portolan.mcp_server.access import set_ai_access
    from tests.verify_seed import TENANT

    conn = db_module.connect(db_url, autocommit=True)
    try:
        set_ai_access(conn, TENANT, enabled=False, changed_by="test", reason="hint check")
        code, result, _path = _run(verify_app, "mcp", tmp_path)
    finally:
        # module-scoped app: reopen the door for the tests that follow
        set_ai_access(conn, TENANT, enabled=True, changed_by="test", reason="hint check done")
        conn.close()

    assert code == 1 and result["ok"] is False
    tools = next(c for c in result["checks"] if c["name"] == "MCP tools/list")
    assert tools["ok"] is False
    assert "-32010" in tools["hint"]
    assert "python -m portolan.mcp_server access on" in tools["hint"]
    out = capsys.readouterr().out
    assert out.startswith("FAIL ")
    assert "hint: mcp: MCP tools/list: AI access is off" in out


def test_report_cards_and_action_note_pass_on_the_seed(
    verify_app: VerifyApp, tmp_path: Path
) -> None:
    """The seed confirms one card after its report was written, so the
    report page must show it confirmed from the queue, and the confirmed
    action must download and show its why."""
    assert verify_app.counts["actions_with_why"] == 1
    for target in ("report-cards", "action-note"):
        code, result, _path = _run(verify_app, target, tmp_path / target)
        assert code == 0, _failed(result)
    by_name = {c["name"]: c for c in result["checks"]}
    md = {m["name"]: m["found"] for m in by_name["action .md download"]["markers"]}
    assert md["evidence lines with a source link"] >= 1
    assert md["boilerplate synthesis"] == 0
    account = by_name["account in Atlas lists its open actions"]["markers"][0]
    assert account["found"] >= 1


def test_a_report_that_ignores_the_queue_fails_report_cards(
    verify_app: VerifyApp, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "portolan.web.routes.reports.refresh_action_card_lines", lambda conn, body, week: body
    )

    code, result, _path = _run(verify_app, "report-cards", tmp_path)

    assert code == 1
    check = next(
        c for c in result["checks"] if c["name"] == "report card states match the review queue"
    )
    bad = {m["name"]: m["found"] for m in check["markers"] if not m["ok"]}
    assert bad == {"card lines out of step with the review queue": 1}


def test_an_action_download_without_its_why_fails_action_note(
    verify_app: VerifyApp, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("portolan.web.routes.notes.action_why_for_note", lambda conn, note: None)

    code, result, _path = _run(verify_app, "action-note", tmp_path)

    assert code == 1
    download = next(c for c in result["checks"] if c["name"] == "action .md download")
    bad = {m["name"] for m in download["markers"] if not m["ok"]}
    assert "Why section" in bad
    assert "boilerplate synthesis" in bad


# ---------------------------------------------------------------------
# The opt-in chat target. The app's own model is the fake client with no
# script, so every test that wants an answer puts `claim_echo_llm` (or a
# canned `Answer`) behind the chat route; no test makes a model call.
# ---------------------------------------------------------------------

LATE_DELIVERIES = "Which hospital accounts had late deliveries in August, and what did they say?"
NPS = "What is our NPS with Ashgrove Health?"


def _stub_chat(monkeypatch, **stub) -> None:
    from tests.verify_seed import answer_with, claim_echo_llm

    monkeypatch.setattr(
        "portolan.web.routes.chat.answer_question", answer_with(claim_echo_llm(**stub))
    )


def _chat_citations(record: dict) -> list[dict]:
    return [c for s in record["sentences"] for c in s["citations"]]


def test_chat_passes_a_cited_answer_and_records_it(
    verify_app: VerifyApp, db_url: str, tmp_path: Path, monkeypatch, capsys
) -> None:
    from tests.verify_seed import STUB_MODEL

    _stub_chat(monkeypatch)

    code, result, path = _run(verify_app, "chat", tmp_path, "--question", LATE_DELIVERIES)

    assert code == 0, _failed(result)
    (record,) = result["chat"]
    assert record["ok"] is True
    assert (record["outcome"], record["model"], record["withheld_sentences"]) == (
        "answered",
        STUB_MODEL,
        0,
    )
    assert 0 < record["first_sentence_ms"] <= record["answer_ms"]
    citations = _chat_citations(record)
    assert citations and all(c["ok"] for c in citations)
    assert all(c["claim_id"] and c["original_href"] for c in citations)
    assert any(c.get("quote_match") in ("exact", "fuzzy") for c in citations)
    assert all(c.get("cited_block_found") is not False for c in citations)
    # Each sentence the stub wrote is one claim's text, word for word: the
    # parsed sentence text is the answer, without the chip labels.
    with psycopg.connect(db_url) as conn:
        for sentence in record["sentences"]:
            (claim_text,) = conn.execute(
                "SELECT claim_text FROM claims WHERE id = %s",
                (sentence["citations"][0]["claim_id"],),
            ).fetchone()
            assert sentence["text"] == " ".join(claim_text.split())
    assert record["answer_text"] == " ".join(s["text"] for s in record["sentences"])
    assert result["derived"]["chat_conversation_id"] == record["conversation_id"]

    written = path.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert verify_app.org_key not in written and verify_app.org_key not in out
    assert out.startswith("PASS ")
    assert f"q1 PASS {record['answer_ms'] / 1000:.1f} s, answered, " in out


def test_chat_fails_a_citation_to_a_missing_note(
    verify_app: VerifyApp, tmp_path: Path, monkeypatch
) -> None:
    from portolan.web.answer import Answer, Citation, Sentence

    answer = Answer(
        sentences=[
            Sentence(
                text="Two shipments arrived late.",
                citations=[Citation(note_slug="signal-no-such-note")],
            )
        ],
        retrieved=[],
        control_ok=True,
        usage={"model": "canned"},
    )
    monkeypatch.setattr("portolan.web.routes.chat.answer_question", lambda *a, **k: answer)

    code, result, _path = _run(verify_app, "chat", tmp_path, "--question", LATE_DELIVERIES)

    assert code == 1
    failed = {c["name"]: c for c in _failed(result)}
    assert set(failed) == {"cited note page", "q1 every citation resolves"}
    assert failed["cited note page"]["status"] == 404
    assert "/notes/signal-no-such-note" in failed["q1 every citation resolves"]["error"]
    (citation,) = _chat_citations(result["chat"][0])
    assert (citation["note_level"], citation["ok"]) == (True, False)


def test_chat_reports_a_sentence_the_number_gate_withheld(
    verify_app: VerifyApp, tmp_path: Path, monkeypatch, capsys
) -> None:
    from tests.verify_seed import UNVERIFIED_NUMBER_SENTENCE

    _stub_chat(monkeypatch, unverified_number=True)

    code, result, _path = _run(verify_app, "chat", tmp_path, "--question", LATE_DELIVERIES)

    # Recorded, not failed: the gate withholding a sentence is the gate working.
    assert code == 0, _failed(result)
    (record,) = result["chat"]
    assert record["withheld_sentences"] == 1
    assert UNVERIFIED_NUMBER_SENTENCE not in record["answer_text"]
    gate = next(
        c for c in result["checks"] if c["name"] == "q1 sentences withheld by the number gate"
    )
    assert (gate["ok"], gate["withheld"]) == (True, 1)
    assert "citations, 1 withheld: " in capsys.readouterr().out


def test_chat_nothing_recorded_fails_a_question_and_passes_expect_nothing(
    verify_app: VerifyApp, tmp_path: Path, monkeypatch
) -> None:
    _stub_chat(monkeypatch, cite_nothing=True)

    code, result, _path = _run(verify_app, "chat", tmp_path / "question", "--question", NPS)
    assert code == 1
    assert [c["name"] for c in _failed(result)] == ["q1 answer carries a citation"]
    assert result["chat"][0]["outcome"] == "nothing_recorded"

    code, result, _path = _run(verify_app, "chat", tmp_path / "expect", "--expect-nothing", NPS)
    assert code == 0, _failed(result)
    assert result["chat"][0]["expect"] == verify.EXPECT_NOTHING


def test_chat_fails_when_the_quote_is_not_on_the_original(
    verify_app: VerifyApp, tmp_path: Path, monkeypatch
) -> None:
    from portolan.web.routes import originals

    real_view = originals._build_view

    def _reworded(source_row, raw_bytes):
        view = real_view(source_row, raw_bytes)
        for block in view.get("blocks", []):
            block["text"] = "nothing the customer said"
        if "text" in view:
            view["text"] = "nothing the customer said"
        return view

    monkeypatch.setattr(originals, "_build_view", _reworded)
    _stub_chat(monkeypatch)

    code, result, _path = _run(verify_app, "chat", tmp_path, "--question", LATE_DELIVERIES)

    assert code == 1
    assert [c["name"] for c in _failed(result)] == ["q1 every citation resolves"]
    quoted = [c for c in _chat_citations(result["chat"][0]) if c["quote"]]
    assert quoted and all(c["quote_match"] == "none" for c in quoted)
    assert "the claim's quote is not on /originals/" in _failed(result)[0]["error"]


def _passage_echo_llm():
    """A chat model stub that answers only from matched passages: each
    sentence is a run of plain words copied from one passage, cited by its
    passage id with that run as the quote."""
    import re

    from portolan.llm.base import LLMResult
    from portolan.llm.fake import FakeLLMClient
    from tests.verify_seed import STUB_MODEL, TENANT

    markup = re.compile(r"\[\[.*?\]\]|\^\[.*?\]|\(.*?\)|[*_#>\"“”:;|]|^- ", re.M)

    def respond(*, user: str, **_kwargs) -> LLMResult:
        sentences = []
        for block in user.split("### Note ")[1:]:
            found = re.search(r'\[passage_id=(p-\w+)\][^\n]*\n"""\n(.*?)\n"""', block, re.S)
            if found is None:
                continue
            slug, (pid, body) = block.split("\n", 1)[0].strip(), found.groups()
            runs = [" ".join(r.split()) for r in markup.split(body)]
            words = max(runs, key=len).split()[:10]
            if len(words) >= 4:
                quote = " ".join(words)
                sentences.append(
                    {
                        "text": quote,
                        "note_slugs": [slug],
                        "claim_ids": [],
                        "passages": [{"passage_id": pid, "quote": quote}],
                    }
                )
        data = {"sentences": sentences[:2], "gaps": []}
        return LLMResult(
            text=json.dumps(data),
            data=data,
            input_tokens=len(user) // 4,
            output_tokens=10,
            cached_tokens=0,
            model=STUB_MODEL,
        )

    return FakeLLMClient(respond, model_key=STUB_MODEL, tenant=TENANT)


def test_chat_passes_a_passage_cited_answer(
    verify_app: VerifyApp, tmp_path: Path, monkeypatch
) -> None:
    from tests.verify_seed import answer_with

    monkeypatch.setattr(
        "portolan.web.routes.chat.answer_question", answer_with(_passage_echo_llm())
    )

    code, result, _path = _run(verify_app, "chat", tmp_path, "--question", NPS)

    assert code == 0, _failed(result)
    (record,) = result["chat"]
    assert record["outcome"] == "answered"
    citations = _chat_citations(record)
    assert citations and all(c["ok"] and c["passage_quote"] for c in citations)
    assert all(c["passage_quote_match"] in ("exact", "fuzzy") for c in citations)
    assert not any(c["note_level"] or c["claim_id"] for c in citations)


def test_chat_fails_when_the_answer_errors(verify_app: VerifyApp, tmp_path: Path) -> None:
    """No stub: the app's fake model has no script and raises, so the
    stream sends its generic error event."""
    code, result, _path = _run(verify_app, "chat", tmp_path, "--question", LATE_DELIVERIES)

    assert code == 1
    arrived = next(c for c in result["checks"] if c["name"] == "q1 answer arrived")
    assert (arrived["ok"], arrived["status"]) == (False, 200)
    assert arrived["error"].startswith("error event: The answer could not be generated")
    assert result["chat"][0]["outcome"] == "error"


def test_chat_questions_come_from_flags_the_file_and_expect_nothing(tmp_path: Path) -> None:
    listed = tmp_path / "questions.txt"
    listed.write_text(f"{LATE_DELIVERIES}\n\n{NPS}\n", encoding="utf-8")

    assert verify.collect_questions(["Q?", LATE_DELIVERIES], listed, [NPS, "Z?"]) == [
        ("Q?", verify.EXPECT_ANSWER),
        (LATE_DELIVERIES, verify.EXPECT_ANSWER),
        (NPS, verify.EXPECT_NOTHING),
        ("Z?", verify.EXPECT_NOTHING),
    ]
    with pytest.raises(SystemExit):
        verify.main(["chat", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        verify.main(["health", "--question", "Q?", "--out-dir", str(tmp_path)])
    assert not list(tmp_path.glob("*.json"))


def test_demo_questions_file_is_the_demo_scripts_ten() -> None:
    import re

    script = require_repo_path("01-spec", "demo-script-2026-09-22.md").read_text(encoding="utf-8")
    written = re.findall(r'^\d+\. \*\*"(.+)"\*\*$', script, flags=re.M)
    assert len(written) == 10
    listed = require_repo_path("scripts", "demo_questions.txt").read_text(encoding="utf-8")
    assert listed.splitlines() == written


def test_chat_constants_match_the_app() -> None:
    from portolan.extract.quote_gate import FUZZY_QUOTE_THRESHOLD, quote_is_supported
    from portolan.web.answer import INDEX_BROKEN_MESSAGE, NOTHING_RECORDED_MESSAGE

    assert NOTHING_RECORDED_MESSAGE.startswith(verify.NOTHING_RECORDED_PREFIX)
    assert INDEX_BROKEN_MESSAGE.startswith(verify.INDEX_BROKEN_PREFIX)
    assert verify.FUZZY_QUOTE_THRESHOLD == FUZZY_QUOTE_THRESHOLD
    text = "Two shipments showed up days late during the week of July twenty-seventh."
    for quote in (
        "Two shipments showed up days late",
        "two shipments  showed up — days late",
        "Two shipments showed up many days late during the week",
        "The pumps were recalled in March",
    ):
        assert (verify.quote_match(quote, text) != "none") == quote_is_supported(quote, text)
