"""The `portolan` CLI (`portolan/cli/`) against a real app: `create_app` under
uvicorn on 127.0.0.1, on the session's test database seeded by
`tests/verify_seed.py` (the seed `tests/test_verify_script.py` uses), with
the claim-echo chat stub and `chat_mode="single"`, so no model is called.

The scenario files in `tests/cli_scripts/` run through `portolan script`,
the same way an agent would run them.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest

from portolan.cli import main
from portolan.cli.core import EXIT_ERROR, EXIT_FAILED, EXIT_REFUSED, EXIT_USAGE
from portolan.cli.expect import ExpectError, check, parse_expect
from tests._paths import needs_repo_path, require_repo_path
from tests.verify_seed import (
    answer_with,
    app_settings,
    claim_echo_llm,
    seed_verify_graph,
    serve,
    write_keys,
)

SCRIPTS = Path(__file__).resolve().parent / "cli_scripts"


@dataclass
class CliApp:
    base_url: str
    org_key: str
    mcp_key: str
    org_key_file: Path
    mcp_key_file: Path


@pytest.fixture(scope="module")
def cli_app(db_url: str, tmp_path_factory: pytest.TempPathFactory):
    try:
        psycopg.connect(db_url).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unreachable at {db_url!r}: {exc}")
    workdir = tmp_path_factory.mktemp("cli-app")
    store_root = workdir / "store"
    seed_verify_graph(db_url, store_root)
    org_key, mcp_key, org_path, mcp_path = write_keys(workdir / "keys")
    settings = app_settings(
        db_url,
        store_root,
        org_key=org_key,
        session_secret=secrets.token_hex(32),
        mcp_org_key=mcp_key,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("portolan.web.routes.chat.answer_question", answer_with(claim_echo_llm()))
        with serve(settings) as base_url:
            yield CliApp(base_url, org_key, mcp_key, org_path, mcp_path)


def run(app: CliApp, session: Path, capsys, *args: str, extra: bool = True) -> tuple[int, str, str]:
    """`portolan <args>` against the app; `(exit code, stdout, stderr)`."""
    argv = list(args)
    if extra:
        argv += [
            "--base-url",
            app.base_url,
            "--org-key-file",
            str(app.org_key_file),
            "--mcp-key-file",
            str(app.mcp_key_file),
            "--session",
            str(session),
        ]
    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


# ---- scenario files -----------------------------------------------------------------


@pytest.mark.parametrize("script", ["smoke.json", "refusals.json"])
def test_the_shipped_scenarios_pass_on_the_seed(cli_app, tmp_path, capsys, script) -> None:
    code, out, err = run(cli_app, tmp_path / "s.json", capsys, "script", str(SCRIPTS / script))

    assert code == 0, out + err
    assert out.strip().splitlines()[-1].startswith("PASS: ")
    assert "FAIL" not in out


def test_a_failed_expectation_exits_1_and_names_the_step(cli_app, tmp_path, capsys) -> None:
    scenario = tmp_path / "bad.json"
    scenario.write_text(
        json.dumps(
            {
                "steps": [
                    {"do": "health", "expect": {"status": "ok"}},
                    {"do": "actions", "expect": {"open": ">=9999"}},
                    {"do": "health"},
                ]
            }
        )
    )

    code, out, _err = run(cli_app, tmp_path / "s.json", capsys, "script", str(scenario), "--json")

    assert code == EXIT_FAILED
    report = json.loads(out)
    assert [s["ok"] for s in report["steps"]] == [True, False]  # stopped at the first failure
    assert "open >= 9999" in report["steps"][1]["error"]


def test_a_step_the_app_refuses_ends_the_script_with_that_exit_code(
    cli_app, tmp_path, capsys
) -> None:
    scenario = tmp_path / "refused.json"
    scenario.write_text(json.dumps({"steps": [{"do": "note no-such-note"}]}))

    code, _out, _err = run(cli_app, tmp_path / "s.json", capsys, "script", str(scenario))

    assert code == EXIT_REFUSED


@needs_repo_path("scripts", "demo_questions.txt")
def test_the_live_demo_scenario_asks_the_ten_demo_questions_and_is_never_run_here() -> None:
    spec = json.loads((SCRIPTS / "demo_live.json").read_text())
    asked = [
        json.loads(step["do"].removeprefix("ask "))
        for step in spec["steps"]
        if step.get("do", "").startswith("ask ")
    ]
    demo = require_repo_path("scripts", "demo_questions.txt").read_text().splitlines()

    assert asked == [line for line in demo if line.strip()]
    assert len(asked) == 10


# ---- commands ------------------------------------------------------------------------


def test_ask_keeps_the_conversation_across_shell_calls_and_cites(cli_app, tmp_path, capsys) -> None:
    session = tmp_path / "s.json"
    assert run(cli_app, session, capsys, "new")[0] == 0

    code, out, _ = run(
        cli_app, session, capsys, "ask", "Which customers are pushing on price?", "--json"
    )
    first = json.loads(out)
    code2, out2, _ = run(
        cli_app, session, capsys, "ask", "Are the delivery problems over?", "--json"
    )
    second = json.loads(out2)
    code3, out3, _ = run(
        cli_app,
        session,
        capsys,
        "ask",
        "--new-conversation",
        "Which customers are pushing on price?",
        "--json",
    )
    third = json.loads(out3)

    assert (code, code2, code3) == (0, 0, 0)
    assert first["outcome"] == "answered" and first["citations"]
    assert all(c["ok"] and c["original"] for c in first["citations"])
    assert first["new_conversation"] is True
    assert second["new_conversation"] is False
    assert second["conversation_id"] == first["conversation_id"]
    assert third["conversation_id"] != first["conversation_id"]
    stored = json.loads(session.read_text())
    assert stored["conversation_id"] == third["conversation_id"]
    assert [entry["exit"] for entry in stored["log"]] == [0, 0, 0, 0]
    assert stored["log"][1]["command"].startswith("ask 'Which customers")


def test_the_session_file_and_all_output_never_hold_a_key(cli_app, tmp_path, capsys) -> None:
    session = tmp_path / "s.json"
    seen = []
    for args in (
        ("new",),
        ("health",),
        ("ask", "Which customers are pushing on price?"),
        ("mcp", "tools"),
        ("mcp", "call", "coverage", '{"since_days": 1}'),
        ("usage", "--json"),
    ):
        code, out, err = run(cli_app, session, capsys, *args)
        assert code == 0, out + err
        seen += [out, err]
    seen.append(session.read_text())

    for text in seen:
        assert cli_app.org_key not in text
        assert cli_app.mcp_key not in text


def test_a_wrong_org_key_is_refused_with_exit_3_and_is_not_echoed(
    cli_app, tmp_path, capsys
) -> None:
    wrong = tmp_path / "wrong-key.txt"
    wrong.write_text("this-is-not-the-org-key\n")
    argv = ["health", "--base-url", cli_app.base_url, "--org-key-file", str(wrong)]

    code = main([*argv, "--session", str(tmp_path / "s.json")])
    captured = capsys.readouterr()

    assert code == EXIT_REFUSED
    assert "sign-in refused" in captured.err
    assert "this-is-not-the-org-key" not in captured.out + captured.err


def test_an_app_that_is_down_exits_4(tmp_path, capsys) -> None:
    code = main(
        ["health", "--base-url", "http://127.0.0.1:9", "--session", str(tmp_path / "s.json")]
    )

    assert code == EXIT_ERROR
    assert "Connect" in capsys.readouterr().err


def test_usage_errors_exit_2(cli_app, tmp_path, capsys) -> None:
    session = tmp_path / "s.json"
    for args in (
        ("nonsense",),
        ("ask",),
        ("search", "x", "--since", "yesterday"),
        ("review", "decide", "1", "maybe", "--yes"),
        ("health", "--base-url", "http://example.com"),  # local only talks to loopback
    ):
        code, out, err = run(cli_app, session, capsys, *args, extra=args[0] != "health")
        assert code == EXIT_USAGE, (args, out, err)


def test_json_output_and_errors_are_machine_readable(cli_app, tmp_path, capsys) -> None:
    session = tmp_path / "s.json"

    code, out, _ = run(
        cli_app, session, capsys, "note", "signal-complaint-northlake-general-hospital", "--json"
    )
    note = json.loads(out)
    refused_code, refused_out, _ = run(cli_app, session, capsys, "note", "no-such-note", "--json")
    refused = json.loads(refused_out)

    assert code == 0 and len(note["claims"]) >= 2
    assert all(c["quote"] and c["original"] for c in note["claims"])
    assert refused_code == EXIT_REFUSED
    assert (refused["exit"], "404" in refused["message"]) == (EXIT_REFUSED, True)


def test_open_checks_a_quote_from_a_note_page_against_its_original(
    cli_app, tmp_path, capsys
) -> None:
    session = tmp_path / "s.json"
    _, out, _ = run(
        cli_app, session, capsys, "note", "signal-complaint-northlake-general-hospital", "--json"
    )
    claim = json.loads(out)["claims"][0]

    code, out, _ = run(
        cli_app, session, capsys, "open", claim["original"], "--quote", claim["quote"], "--json"
    )
    found = json.loads(out)
    miss_code, miss_out, _ = run(
        cli_app,
        session,
        capsys,
        "open",
        claim["original"],
        "--quote",
        "words that appear nowhere xyzzy",
        "--json",
    )

    assert code == 0 and found["quote_match"] in ("exact", "fuzzy") and found["block_found"] is True
    assert miss_code == EXIT_FAILED and json.loads(miss_out)["quote_match"] == "none"


def test_a_local_report_module_runs_and_a_remote_target_is_a_usage_error(
    cli_app, tmp_path, capfd
) -> None:
    code = main(["report", "weekly", "--", "--help"])
    out = capfd.readouterr().out
    remote = main(["report", "weekly", "--target", "https://portolan.example", "--", "--help"])
    capfd.readouterr()

    assert code == 0 and "--as-of" in out
    assert remote == EXIT_USAGE


def test_the_default_session_path_and_environment_override(
    cli_app, tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PORTOLAN_SESSION", str(tmp_path / "from-env.json"))

    code, _out, _err = run(
        cli_app,
        tmp_path / "unused.json",
        capsys,
        "new",
        extra=False,
    )

    assert code == 0
    assert (tmp_path / "from-env.json").is_file()
    assert not (tmp_path / ".portolan-session.json").exists()


def test_review_decide_needs_yes_and_writes_once(cli_app, tmp_path, capsys) -> None:
    """Last in the file: it closes a review row in the module's shared database."""
    session = tmp_path / "s.json"
    _, out, _ = run(cli_app, session, capsys, "review", "list", "--json")
    before = json.loads(out)
    row = before["cards"][0]["id"]

    no_yes, _, err = run(cli_app, session, capsys, "review", "decide", str(row), "reject")
    code, out, _ = run(
        cli_app,
        session,
        capsys,
        "review",
        "decide",
        str(row),
        "reject",
        "--yes",
        "--reason",
        "cli test",
        "--json",
    )
    decided = json.loads(out)
    again_code, out, _ = run(
        cli_app, session, capsys, "review", "decide", str(row), "reject", "--yes", "--json"
    )
    again = json.loads(out)
    _, out, _ = run(cli_app, session, capsys, "review", "list", "--json")
    after = json.loads(out)

    assert no_yes == EXIT_USAGE and "--yes" in err
    assert (code, decided["status"], decided["already_handled"]) == (0, 303, False)
    assert (again_code, again["already_handled"]) == (0, True)
    assert after["open"] == before["open"] - 1
    assert row not in [c["id"] for c in after["cards"]]


# ---- expect ---------------------------------------------------------------------------


def test_expect_accepts_the_three_spellings() -> None:
    data = {"outcome": "answered", "citations": [1, 2], "cited_notes": ["signal-a", "x"], "n": 3}

    as_object = check(
        parse_expect(
            {"outcome": "answered", "citations": ">=2", "cited_notes": "contains signal-"}
        ),
        data,
    )
    as_list = check(parse_expect(["citations>=2", "n == 3", "outcome!=nothing"]), data)
    as_string = check(parse_expect("n<4"), data)

    assert all(r["ok"] for r in as_object + as_list + as_string)


def test_expect_operators_fail_when_they_should() -> None:
    data = {"outcome": "answered", "citations": [1], "n": 3, "flag": False}
    cases = {
        "citations>=2": False,
        "citations>1": False,
        "citations<=1": True,
        "n<3": False,
        "n>2": True,
        "n!=3": False,
        "outcome==answered": True,
        "outcome contains NSWER": True,
        "flag==false": True,
        "missing==1": False,
    }

    results = {text: check(parse_expect(text), data)[0]["ok"] for text in cases}

    assert results == cases


def test_expect_names_a_metric_that_is_not_there_and_rejects_nonsense() -> None:
    (result,) = check(parse_expect({"nope": 1}), {"yes": 1})

    assert (result["ok"], result["error"]) == (False, "no such metric")
    with pytest.raises(ExpectError):
        parse_expect(["not a condition"])
    with pytest.raises(ExpectError):
        parse_expect([1])
