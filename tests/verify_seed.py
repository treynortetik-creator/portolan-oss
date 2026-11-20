"""A seeded, running app for `scripts/verify.py` to check.

Not a `test_*.py` module (mirrors `tests/merge_helpers.py`): imported by
`tests/test_verify_script.py`, and runnable on its own for the local proof
run:

    uv run python -m tests.verify_seed --database-url \\
        postgresql://localhost/portolan_test_verify --port 8765

which rebuilds that database, seeds it, writes an org key and an MCP key to
files in `--workdir` (a fresh temp directory by default; the paths are
printed, the keys never are), and serves the app on 127.0.0.1 until Ctrl-C.
Then, in a second shell:

    uv run python scripts/verify.py all --base-url http://127.0.0.1:8765 \\
        --org-key-file <workdir>/org-key.txt --mcp-key-file <workdir>/mcp-key.txt

The seed is the web tests' own (`tests/test_web.py` `demo_conn`): one real
`pipeline.run` with the fake extractor over `corpus/out`, which gives
accounts, signals, owned actions with due dates and open review rows. What
the demo graph lacks is added through the real code paths: two action cards
queued for 2026-W36 (`researcher.queue_action_card`, each an open review
row), a real weekly report for that week (`weekly.build_weekly` with the
report tests' canned model, `tests/test_report_helpers.build_report_llm`),
so the report carries action-card lines under "Suggested actions", and then
the second card confirmed (`_confirm_card_after_the_report`), so one action
has a researched why and the stored report still reads "proposed" for it.

`--chat-stub` answers chat with `claim_echo_llm` instead of the fake
client's empty script (which fails every question), so `verify.py chat` can
be proved locally with no model call:

    uv run python scripts/verify.py chat --base-url http://127.0.0.1:8765 \\
        --org-key-file <workdir>/org-key.txt --questions-file scripts/demo_questions.txt
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn

from portolan import db as db_module
from portolan.config import Settings
from portolan.connectors.file_connector import FileConnector
from portolan.fake_extractor import FakePlacements, build_fake_llm_client
from portolan.llm.base import LLMResult
from portolan.llm.fake import FakeLLMClient
from portolan.mcp_server.access import set_ai_access
from portolan.pipeline import run as pipeline_run
from portolan.registry import Registry
from portolan.report import weekly
from portolan.report.researcher import CURRENT_BRIEF_VERSION, ActionCard, queue_action_card
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.util.redact import redact_dsn
from portolan.web.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_OUT = REPO_ROOT / "corpus" / "out"
TENANT = "verify"
REPORT_WEEK = "2026-W36"
# A Friday inside 2026-W36; `build_weekly(week=None, as_of=...)` derives
# the week from it, the same way `tests/test_report_weekly.py` does.
REPORT_AS_OF = date(2026, 9, 4)
CARD_WHAT = "Send a written root-cause summary for the late deliveries."
CONFIRMED_CARD_WHAT = "Call the account about the second late shipment."
CONFIRMED_BY = "verify seed"


def _database_name(url: str) -> str:
    return urlsplit(url).path.rsplit("/", 1)[-1].split("?", 1)[0]


def require_test_database(url: str) -> None:
    """The seed drops the public schema, so it only ever runs against a
    database whose name says it is a test database."""
    name = _database_name(url)
    if not name.startswith("portolan_test"):
        raise ValueError(
            f"refusing to seed {name!r}: the name must start with 'portolan_test' "
            "(the seed drops and rebuilds the public schema)"
        )


def app_settings(
    database_url: str,
    store_root: Path,
    *,
    org_key: str,
    session_secret: str,
    mcp_org_key: str | None,
) -> Settings:
    return Settings(
        database_url=database_url,
        tenant=TENANT,
        store_root=store_root,
        llm_provider="fake",
        embed_provider="fake",
        org_key=org_key,
        session_secret=session_secret,
        mcp_org_key=mcp_org_key,
        # The app is served over plain http://127.0.0.1; a Secure cookie
        # would never come back (`Settings.cookie_secure`'s own comment).
        cookie_secure=False,
        # The stub chat model answers one call; agent mode is covered in
        # tests/test_chat_agent.py.
        chat_mode="single",
    )


STUB_MODEL = "verify-chat-stub"
UNVERIFIED_NUMBER_SENTENCE = "That came to 987654 late shipments in all."
# One claim line of `portolan.web.answer._context_block`'s prompt.
_PROMPT_CLAIM_RE = re.compile(r"^- \[claim_id=(\d+)\] (.+) \(source: \S+ @ .+\)$")


def claim_echo_llm(
    *, claims: int = 2, unverified_number: bool = False, cite_nothing: bool = False
) -> FakeLLMClient:
    """A chat model stub: each sentence is one claim from the notes in the
    prompt, word for word, citing that claim, so the real citation check
    and number gate pass it. `unverified_number` adds a sentence the number
    gate must withhold; `cite_nothing` answers with no sentence at all."""

    def respond(*, user: str, **_kwargs) -> LLMResult:
        sentences: list[dict] = []
        slug = None
        for line in user.splitlines():
            if line.startswith("### Note "):
                slug = line.removeprefix("### Note ").strip()
                continue
            match = _PROMPT_CLAIM_RE.match(line)
            if match and slug and len(sentences) < claims and not cite_nothing:
                claim_id, claim_text = int(match.group(1)), match.group(2)
                sentences.append(
                    {"text": claim_text, "note_slugs": [slug], "claim_ids": [claim_id]}
                )
        if unverified_number and sentences:
            sentences.append({**sentences[0], "text": UNVERIFIED_NUMBER_SENTENCE})
        data = {"sentences": sentences}
        text = json.dumps(data)
        return LLMResult(
            text=text,
            data=data,
            input_tokens=len(user) // 4,
            output_tokens=len(text) // 4,
            cached_tokens=0,
            model=STUB_MODEL,
        )

    return FakeLLMClient(respond, model_key=STUB_MODEL, tenant=TENANT)


def answer_with(llm):
    """A stand-in for `portolan.web.routes.chat.answer_question` that runs
    the real one (search, citation check, number gate) with `llm` in place
    of the app's model."""
    from portolan.web.answer import answer_question

    def _answer(conn, _app_llm, embedder, question, **kwargs):
        return answer_question(conn, llm, embedder, question, **kwargs)

    return _answer


def _confirm_card_after_the_report(conn, what: str) -> int:
    """Confirm the queued card `what` the way `/review`'s Confirm does
    (`confirm_action_card`, then the row resolved with the new action's id
    stamped), after the report was written: the stored report text still
    says "proposed", so `verify.py report-cards` proves the page reads the
    queue at view time, and `verify.py action-note` has a researched why."""
    from portolan.report.researcher import confirm_action_card_with_reason
    from portolan.web.routes.review import _resolve_row

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, payload -> 'card' FROM review_queue "
            "WHERE kind = 'action_card' AND payload -> 'card' ->> 'what' = %s",
            (what,),
        )
        row_id, card = cur.fetchone()
    outcome = confirm_action_card_with_reason(
        conn, card=card, editor=CONFIRMED_BY, as_of=REPORT_AS_OF, review_row_id=row_id
    )
    action_id = outcome.action_id
    if action_id is None:
        raise RuntimeError(f"the seeded action card did not confirm: {outcome.reason}")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE review_queue SET payload = "
            "jsonb_set(payload, '{action_id}', to_jsonb(%s::bigint)) WHERE id = %s",
            (action_id, row_id),
        )
    _resolve_row(conn, row_id, "resolved", CONFIRMED_BY)
    return action_id


def _queue_card(conn, what: str = CARD_WHAT) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.claim_text, c.quote, c.source_slug, c.locator, c.note_slug, l.to_slug
            FROM claims c
            JOIN notes n ON n.slug = c.note_slug AND n.type = 'signal'
            JOIN links l ON l.from_slug = c.note_slug
            JOIN notes acct ON acct.slug = l.to_slug AND acct.type = 'account'
            WHERE c.archived = false AND c.superseded_at IS NULL
            ORDER BY c.id
            LIMIT 1
            """
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("the demo graph has no signal claim linked to an account")
    claim_id, claim_text, quote, source_slug, locator, note_slug, account_slug = row
    card = ActionCard(
        run_id="verify-seed",
        # One open card per candidate (`queue_action_card`): the second card
        # needs a candidate of its own.
        candidate_slug=note_slug if what == CARD_WHAT else f"{note_slug}-second",
        brief_version=CURRENT_BRIEF_VERSION,
        account=account_slug,
        verdict="action",
        what=what,
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
        owner="sam-okafor",
        due=date(2026, 9, 11),
        due_rule="next_report",
        theme=None,
        confidence="medium",
        confidence_refs={"evidence_refs": [0], "finding_refs": []},
    )
    return queue_action_card(conn, card, week=REPORT_WEEK)


def seed_verify_graph(database_url: str, store_root: Path) -> dict[str, int]:
    """Rebuild `database_url`'s schema and seed it. Returns row counts, so a
    caller can see the seed did the work rather than trusting exit 0."""
    from tests.test_report_helpers import build_report_llm

    require_test_database(database_url)
    store_root.mkdir(parents=True, exist_ok=True)
    conn = db_module.connect(database_url)
    try:
        with conn.cursor() as cur:
            # This session only: the pipeline commits once per row, and a
            # throwaway test database does not need each commit flushed
            # to disk before the next one starts.
            cur.execute("SET synchronous_commit = off")
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
        conn.commit()
        db_module.apply_migrations(conn)
        conn.commit()
        conn.autocommit = True

        settings = app_settings(
            database_url, store_root, org_key="seed", session_secret="seed", mcp_org_key=None
        )
        pipeline_run(
            conn,
            connector=FileConnector(CORPUS_OUT),
            llm=build_fake_llm_client(FakePlacements(CORPUS_OUT), tenant=TENANT),
            embedder=FakeEmbedder(),
            registry=Registry.load(conn),
            store=LocalStore(store_root),
            settings=settings,
        )
        _queue_card(conn)
        _queue_card(conn, CONFIRMED_CARD_WHAT)
        weekly.build_weekly(conn, build_report_llm(), FakeEmbedder(), as_of=REPORT_AS_OF)
        _confirm_card_after_the_report(conn, CONFIRMED_CARD_WHAT)
        # Migration 0031 leaves a new database's MCP door closed by design; the seeded
        # graph is what `verify.py mcp` checks, so open it for this tenant here.
        set_ai_access(
            conn,
            TENANT,
            enabled=True,
            changed_by="verify_seed",
            reason="seeded verify graph: verify.py's mcp target lists tools",
        )

        counts: dict[str, int] = {}
        with conn.cursor() as cur:
            for name, sql in (
                ("notes", "SELECT count(*) FROM notes"),
                (
                    "open_actions_with_owner_and_due",
                    "SELECT count(*) FROM actions WHERE status = 'open' "
                    "AND owner_entity_id IS NOT NULL AND due IS NOT NULL",
                ),
                ("open_review_rows", "SELECT count(*) FROM review_queue WHERE status = 'open'"),
                ("reports", "SELECT count(*) FROM reports"),
                ("coverage_rows", "SELECT count(*) FROM coverage_ledger"),
                ("actions_with_why", "SELECT count(*) FROM action_why"),
            ):
                cur.execute(sql)
                counts[name] = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM ai_access WHERE tenant = %s AND enabled", (TENANT,))
            counts["ai_access_open"] = cur.fetchone()[0]
        return counts
    finally:
        conn.close()


@contextmanager
def serve(settings: Settings, *, port: int = 0) -> Iterator[str]:
    """`create_app(settings)` under a real uvicorn on 127.0.0.1, in a
    background thread; yields the base URL. Port 0 picks a free one."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    bound_port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(settings), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            sock.close()
            raise RuntimeError("the app did not start (see the uvicorn log above)")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{bound_port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30)
        sock.close()


def write_keys(workdir: Path) -> tuple[str, str, Path, Path]:
    """Fresh random keys, written to `workdir` with mode 600. Returns the
    two values (for the app's settings) and the two file paths."""
    workdir.mkdir(parents=True, exist_ok=True)
    org_key = secrets.token_hex(24)
    mcp_key = secrets.token_hex(24)
    org_path = workdir / "org-key.txt"
    mcp_path = workdir / "mcp-key.txt"
    for path, value in ((org_path, org_key), (mcp_path, mcp_key)):
        path.write_text(value + "\n", encoding="utf-8")
        path.chmod(0o600)
    return org_key, mcp_key, org_path, mcp_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database-url", default="postgresql://localhost/portolan_test_verify")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument(
        "--chat-stub", action="store_true", help="answer chat with claim_echo_llm (no model call)"
    )
    args = parser.parse_args(argv)

    try:
        require_test_database(args.database_url)
    except ValueError as exc:
        parser.error(str(exc))
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="portolan-verify-"))
    store_root = workdir / "store"
    counts = seed_verify_graph(args.database_url, store_root)
    org_key, mcp_key, org_path, mcp_path = write_keys(workdir)
    settings = app_settings(
        args.database_url,
        store_root,
        org_key=org_key,
        session_secret=secrets.token_hex(32),
        mcp_org_key=mcp_key,
    )
    print(f"seeded {redact_dsn(args.database_url)}: {counts}")
    print(f"org key file: {org_path}")
    print(f"mcp key file: {mcp_path}")
    if args.chat_stub:
        from portolan.web.routes import chat as chat_routes

        chat_routes.answer_question = answer_with(claim_echo_llm())
        print(f"chat answers from the claim-echo stub ({STUB_MODEL}), no model call")
    with serve(settings, port=args.port) as base_url:
        print(f"serving {base_url} (Ctrl-C to stop)", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
