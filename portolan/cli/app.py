"""The parser, the exit-code mapping and `main`."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from portolan.cli.commands import Command, Result, register
from portolan.cli.core import (
    EXIT_ERROR,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_USAGE,
    GLOBAL_OPTIONS,
    CliError,
    Ctx,
    UsageError,
)
from portolan.cli.script import Outcome, load_scenario, render_report, run_scenario
from portolan.client.http import AppFailed, AppRefused
from portolan.client.mcp import AI_ACCESS_CLOSED_CODE, McpError

# JSON-RPC codes for a request the server could not answer, not one it refused.
_JSONRPC_SERVER_ERRORS = {-32603, -32000, -32001, -32002}


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(f"{message} (try: {self.prog} --help)")


def _common() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    hide = argparse.SUPPRESS
    common.add_argument("--json", action="store_true", default=hide, help="machine-readable output")
    common.add_argument(
        "--session",
        default=hide,
        help="session file (default $PORTOLAN_SESSION or ./.portolan-session.json)",
    )
    common.add_argument("--target", default=hide, help="'local' (default) or a base URL")
    common.add_argument(
        "--base-url",
        dest="base_url",
        default=hide,
        help="the local app's URL (default http://127.0.0.1:8000)",
    )
    common.add_argument(
        "--editor",
        default=hide,
        help="the name the app records for this user (default portolan-cli)",
    )
    common.add_argument(
        "--org-key-file", dest="org_key_file", default=hide, help="file holding the org key"
    )
    common.add_argument(
        "--mcp-key-file", dest="mcp_key_file", default=hide, help="file holding the MCP key"
    )
    common.add_argument(
        "--timeout", type=float, default=hide, help="seconds per request (default 30)"
    )
    common.add_argument(
        "--chat-timeout",
        dest="chat_timeout",
        type=float,
        default=hide,
        help="seconds to wait for an answer (default 180)",
    )
    return common


def cmd_script(ctx: Ctx, args: argparse.Namespace) -> Result:
    spec = load_scenario(Path(args.file))
    code, report = run_scenario(
        spec, lambda tokens: invoke(ctx, tokens), restart=not args.keep_session
    )
    result = Result(report, code)
    if code != 0:
        result.message = next((r["error"] for r in report["steps"] if not r["ok"]), "script failed")
    return result


def _render_script(report: dict[str, Any]) -> str:
    return render_report(report)


def build_parser() -> Parser:
    common = _common()
    parser = Parser(
        prog="portolan",
        parents=[common],
        description="Use and test a running Portolan app from the shell. Exit codes: "
        "0 ok, 1 assertion or validation failure, 2 usage error, "
        "3 the app refused, 4 the app errored or timed out.",
    )
    sub = parser.add_subparsers(dest="top", required=True, parser_class=Parser)
    register(sub, common)
    p = sub.add_parser(
        "script", parents=[common], help="run a scenario file with expect assertions"
    )
    p.set_defaults(command=Command(cmd_script, _render_script))
    p.add_argument("file", help="a JSON scenario (see docs/CLI.md)")
    p.add_argument("--keep-session", action="store_true", help="do not start a fresh session first")
    return parser


_PARSER: Parser | None = None


def parser() -> Parser:
    global _PARSER
    if _PARSER is None:
        _PARSER = build_parser()
    return _PARSER


def _mcp_exit(exc: McpError) -> int:
    if exc.status is not None:
        return EXIT_REFUSED if exc.status < 500 else EXIT_ERROR
    if exc.code is None:
        return EXIT_ERROR
    if exc.code == AI_ACCESS_CLOSED_CODE:
        return EXIT_REFUSED
    return EXIT_ERROR if exc.code in _JSONRPC_SERVER_ERRORS else EXIT_REFUSED


def run_namespace(ctx: Ctx, ns: argparse.Namespace, tokens: list[str]) -> Outcome:
    command: Command = ns.command
    render = command.render
    try:
        result = command.handler(ctx, ns)
        outcome = Outcome(result.data, result.exit, result.message, render)
    except UsageError as exc:
        outcome = Outcome(None, EXIT_USAGE, str(exc))
    except CliError as exc:
        outcome = Outcome(None, exc.code, str(exc))
    except AppRefused as exc:
        outcome = Outcome(None, EXIT_REFUSED, str(exc))
    except AppFailed as exc:
        outcome = Outcome(None, EXIT_ERROR, str(exc))
    except McpError as exc:
        outcome = Outcome(None, _mcp_exit(exc), f"MCP: {exc}")
    if ctx.session is not None:
        ctx.session.log(shlex.join(tokens), outcome.code)
    return outcome


def invoke(ctx: Ctx, tokens: list[str]) -> Outcome:
    """One command from tokens, as a script step runs it. Global options
    belong to the outer `portolan script` call."""
    try:
        ns = parser().parse_args(tokens)
    except UsageError as exc:
        return Outcome(None, EXIT_USAGE, str(exc))
    if ns.command.handler is cmd_script:
        return Outcome(None, EXIT_USAGE, "a script cannot run another script")
    return run_namespace(ctx, ns, tokens)


def _emit(outcome: Outcome, as_json: bool) -> None:
    if as_json:
        body: dict[str, Any] = dict(outcome.data or {})
        if outcome.code != EXIT_OK:
            body["exit"] = outcome.code
            body["message"] = outcome.message
        print(json.dumps(body, indent=2, default=str))
        return
    if outcome.data is not None and outcome.render is not None:
        print(outcome.render(outcome.data))
    if outcome.code != EXIT_OK and outcome.message:
        print(f"portolan: {outcome.message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in tokens
    try:
        ns = parser().parse_args(tokens)
    except UsageError as exc:
        _emit(Outcome(None, EXIT_USAGE, str(exc)), as_json)
        return EXIT_USAGE
    options = {name: getattr(ns, name, None) for name in GLOBAL_OPTIONS}
    try:
        ctx = Ctx(options)
    except CliError as exc:
        _emit(Outcome(None, exc.code, str(exc)), as_json)
        return exc.code
    try:
        outcome = run_namespace(ctx, ns, tokens)
    except KeyboardInterrupt:
        outcome = Outcome(None, EXIT_FAILED, "interrupted")
    finally:
        ctx.close()
    _emit(outcome, ctx.json)
    return outcome.code
