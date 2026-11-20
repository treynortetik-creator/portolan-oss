"""Shared fixtures for `tests/test_report_*.py` (wave C1, `portolan.report`).

Not itself a test module (no `test_*` functions) -- imported by the actual
test files for one expensive, session-scoped graph build, mirroring
`tests/test_pipeline.py`'s own `full_run_report_and_db_url` pattern (see
that module's docstring) rather than depending on the shared, function-
scoped `db` fixture in `tests/conftest.py`.

**Own test database**: `postgresql://localhost/portolan_test_report`
(`createdb portolan_test_report` once, per `01-spec/build-contracts.md`
"Own test database"), overridable with `DATABASE_URL`.

**No more `FakePlacements` subclass** (fix wave 2026-09-12). This module
used to wrap `portolan.fake_extractor.FakePlacements` in a local
`_ActionInjectingPlacements` because the real one never populated
`ExtractionResult.actions` at all -- so the truth table's own "open actions
include the F09 one with owner and due date" check would have had nothing
to find. `portolan/fake_extractor.py` is now owned by this fix wave too
(review finding: "fake_extractor never emits actions, never verdicts
contradicts"), and its `extraction_result_for` emits the F09/F32/F33
actions itself, straight off `expectations.report.actions_open` --
identical output to what the subclass used to bolt on, so a plain
`build_fake_llm_client(FakePlacements(...))` is enough here now.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import psycopg
import pytest

from portolan import db as db_module
from portolan.config import Settings
from portolan.connectors.file_connector import FileConnector
from portolan.fake_extractor import FakePlacements, build_fake_llm_client, minimal_json_for_schema
from portolan.llm.fake import FakeLLMClient
from portolan.pipeline import RunReport, run
from portolan.registry import Registry
from portolan.report.accounts import account_arr_by_note_slug, load_account_arr_via_connector
from portolan.report.executive import EXECUTIVE_SYSTEM
from portolan.report.questions import STANDING_QUESTION_SYSTEM
from portolan.report.synthesis import ACCOUNT_SYNTHESIS_SYSTEM
from portolan.report.themes import THEME_NAMING_SYSTEM
from portolan.report.weekly import HEADLINE_SYSTEM
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_OUT = REPO_ROOT / "corpus" / "out"
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"
MANIFEST_PATH = CORPUS_OUT / "manifest.json"

DEFAULT_REPORT_DATABASE_URL = "postgresql://localhost/portolan_test_report"


def _settings(store_root: Path, tenant: str = "report-test") -> Settings:
    return Settings(
        database_url="postgresql://localhost/unused",
        tenant=tenant,
        store_root=store_root,
    )


# --------------------------------------------------------------------------
# A second `LLMClient`, used only for the `portolan.report.*` calls
# themselves (theme naming, standing-question answers, the weekly headline,
# account synthesis) -- kept separate from the `build_fake_llm_client(...)`
# one that builds the graph (pipeline extraction/merge) because those two
# jobs need entirely different canned shapes. Each surface is matched on
# its exact `system` prompt constant (imported straight from the module
# that owns it, `portolan.report.themes.THEME_NAMING_SYSTEM` etc.) rather
# than by guessing at prompt text, so a change to any of those prompts
# would fail loudly here instead of silently matching the wrong branch.
# --------------------------------------------------------------------------

# Anchored to a whole line: `_claims_context` (portolan/report/questions.py)
# renders each retrieved note as a `[[slug]]` header line followed by its
# claim lines, and those claim lines carry their *own* inline `[[slug]]`
# mentions (`link_mentions`, portolan/notes/render.py) -- e.g. a
# `signal-competitor_mention-meadowbrook-community-hospital` note's claim
# line mentions `[[meadowbrook-community-hospital]]` inline, a DIFFERENT
# slug that was never itself retrieved. An unanchored scan over the whole
# prompt would pick up both indiscriminately and could cite a slug the
# search never returned -- exactly the case `answer_standing_questions`
# is supposed to catch and drop. Anchoring to a whole line matches only
# the header slugs `_claims_context` actually retrieved.
_WIKILINK_IN_PROMPT_RE = re.compile(r"^\[\[([a-z0-9][a-z0-9-]*)\]\]$", re.MULTILINE)
_CITATION_IN_PROMPT_RE = re.compile(r"\^\[\s*([^\]@]+?)\s*@\s*([^\]]*)\]")

THEME_NAME = "Late deliveries, Midwest"
THEME_SYNTHESIS = (
    "Hospitals across the Midwest, served by the Dayton and Columbus distribution "
    "branches, reported Tidewater consumable and pump deliveries arriving several "
    "days late through the summer of 2026. The pattern escalated across accounts "
    "and resurfaced as a renewal-time concern at Ashgrove Health."
)


def _report_llm_responder(*, system, user, json_schema, model, surface):
    del model
    if surface == "report" and system == THEME_NAMING_SYSTEM:
        return {"name": THEME_NAME, "synthesis": THEME_SYNTHESIS}

    if surface == "report" and system == STANDING_QUESTION_SYSTEM:
        slugs = list(dict.fromkeys(_WIKILINK_IN_PROMPT_RE.findall(user)))[:3]
        if not slugs:
            return {
                "sentences": [{"text": "No retrieved note answers this question.", "cites": []}]
            }
        text = (
            "The notes touching this window that bear on this question are "
            + ", ".join(slugs)
            + "."
        )
        return {"sentences": [{"text": text, "cites": slugs}]}

    if surface == "report" and system == HEADLINE_SYSTEM:
        return {
            "sentences": [
                "The graph updated across several accounts this window.",
                "See the themes section below for the largest recurring pattern.",
                "Standing questions were answered from notes retrieved this window only.",
                "Outliers this window are listed for review below.",
                "Quiet accounts are flagged below for outreach.",
            ]
        }

    if surface == "consolidate" and system == ACCOUNT_SYNTHESIS_SYSTEM:
        markers = _CITATION_IN_PROMPT_RE.findall(user)[:2]
        citations = " ".join(f"^[{slug} @ {locator}]" for slug, locator in markers)
        text = "This account has open items on file." + (f" {citations}" if citations else "")
        return {"synthesis": text}

    if surface == "report" and system == EXECUTIVE_SYSTEM:
        # Weekly-report wave 2: a real citation marker and a real account
        # slug pulled straight out of THIS call's own payload -- the same
        # "match the prompt's own data, don't hand-roll a canned answer
        # that might not survive the mechanical gates" approach the
        # ACCOUNT_SYNTHESIS_SYSTEM branch above already uses.
        payload = json.loads(user)
        markers = _CITATION_IN_PROMPT_RE.findall(user)[:1]
        citation = f"^[{markers[0][0]} @ {markers[0][1]}]" if markers else ""
        accounts = payload.get("accounts") or []
        by_account = (
            [
                {
                    "account_slug": accounts[0]["slug"],
                    "paragraph": "This account has activity worth watching this week.",
                }
            ]
            if accounts
            else []
        )
        attention = [f"An account raised a concern worth reviewing. {citation}"] if citation else []
        return {
            "this_week": ["Activity touched several accounts across the graph this week."],
            "attention": attention,
            "by_account": by_account,
            "suggested_actions": ["Follow up with the account team on open items."],
        }

    if json_schema is not None:
        return minimal_json_for_schema(json_schema)
    return "ok"


def build_report_llm() -> FakeLLMClient:
    return FakeLLMClient(_report_llm_responder)


class FullGraph:
    """Everything a `test_report_*.py` module needs: an open connection to
    the fully-built graph, the settings/connector/llm/embedder that built
    it, the `RunReport` from that one `portolan.pipeline.run` call, and
    `baseline_lint_errors` -- the lint's error-severity findings captured
    right after that pipeline run, before any `portolan.report` code has
    touched the graph.

    Before the fix wave 2026-09-12 review (G3, G6), `run_lint` on this
    corpus reported 5 pre-existing errors here even with zero report code
    involved -- two `orphan` contact notes (`maria-chen`, `tom-ridley`) and
    three `orphan` `signal-competitor_mention-*` notes (`WIKILINK_RE`
    rejecting the `_` in `competitor_mention`/`contact_change` slugs, and
    contact notes getting no link to their account). Both are fixed now
    (`portolan/notes/render.py`, `portolan/notes/persist.py`), so this
    baseline is expected to be empty on a clean run -- kept as a live
    baseline (diffed, not asserted to be exactly one shape) rather than
    deleted outright, so a *future* regression that reintroduces a
    pre-existing-graph lint error is still caught the same way a
    report-code-introduced one is: `test_report_weekly.py`'s "`run_lint`
    reports zero errors" check is "zero *new* errors relative to this
    baseline".
    """

    def __init__(
        self,
        conn,
        settings,
        connector,
        llm,
        embedder,
        run_report: RunReport,
        baseline_lint_errors: list,
        database_url: str,
    ) -> None:
        self.conn = conn
        self.settings = settings
        self.connector = connector
        self.llm = llm
        self.embedder = embedder
        self.run_report = run_report
        self.baseline_lint_errors = baseline_lint_errors
        # The *real* database this graph lives in -- `settings.database_url`
        # is a dummy placeholder (`portolan.pipeline.run` never reads it;
        # it's handed an already-open `conn`), so a test that needs to open
        # a second connection to this same graph (the CLI subprocess test)
        # or spawn a subprocess against it needs this instead.
        self.database_url = database_url


def _session_database_url(base_url: str) -> str:
    """`base_url` with `_report_session` appended to the database name.

    `tests/conftest.py`'s function-scoped `db` fixture drops and rebuilds
    the *entire* `public` schema of whatever `DATABASE_URL` names, once per
    test, for every other module in this suite. Running the whole
    `tests/` directory in one pytest process with `DATABASE_URL` set to
    this wave's own `portolan_test_report` would otherwise point both that
    fixture and this module's session-scoped `full_graph` at the exact
    same schema -- the first non-report test using `db` wipes the graph
    out from under every `test_report_*.py` test that runs afterward in
    the same session. Deriving a second, differently-named database from
    whatever `DATABASE_URL` says keeps this wave's graph on physically
    separate storage `tests/conftest.py` never touches, whatever
    `DATABASE_URL` happens to be set to for the run.
    """
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(base_url)
    base_name = parts.path.lstrip("/") or "portolan_test_report"
    new_path = f"/{base_name}_report_session"
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))


def _ensure_database_exists(url: str) -> None:
    """`CREATE DATABASE` for `url`'s dbname if it doesn't exist yet,
    connecting to the server's `postgres` maintenance database to do it
    (you cannot run `CREATE DATABASE` against the database you're
    creating)."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    dbname = parts.path.lstrip("/")
    maintenance_url = urlunsplit((parts.scheme, parts.netloc, "/postgres", "", ""))

    conn = psycopg.connect(maintenance_url, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{dbname}"')
    finally:
        conn.close()


@pytest.fixture(scope="session")
def full_graph(tmp_path_factory):
    import os

    configured_url = os.environ.get("DATABASE_URL", DEFAULT_REPORT_DATABASE_URL)
    database_url = _session_database_url(configured_url)
    try:
        _ensure_database_exists(database_url)
        conn = db_module.connect(database_url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unreachable at {database_url!r}: {exc}")
        return

    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE")
        cur.execute("CREATE SCHEMA public")
    conn.commit()
    db_module.apply_migrations(conn)
    conn.autocommit = True

    tmp_path = tmp_path_factory.mktemp("report_e2e")
    settings = _settings(tmp_path / "store")
    connector = FileConnector(CORPUS_OUT)
    placements = FakePlacements(CORPUS_OUT)
    llm = build_fake_llm_client(placements, tenant=settings.tenant)
    embedder = FakeEmbedder()
    store = LocalStore(settings.store_root)
    registry = Registry.load(conn)

    run_report = run(
        conn,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
    )

    from portolan.lint.report import run_lint

    baseline_lint_errors = run_lint(conn, store).errors()

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO standing_questions (reader, question) VALUES (%s, %s)",
            ("ops", "Which accounts mentioned a competitor this week?"),
        )

    graph = FullGraph(
        conn, settings, connector, llm, embedder, run_report, baseline_lint_errors, database_url
    )
    yield graph
    conn.close()


@pytest.fixture(scope="session")
def report_llm() -> FakeLLMClient:
    return build_report_llm()


@pytest.fixture(scope="session")
def arr_by_note_slug(full_graph: FullGraph) -> dict[str, float]:
    """`{account_note_slug: arr}` for the whole corpus, read from
    `corpus/out/crm/accounts.csv` via the same connector the graph was
    built from (`portolan.report.accounts` module docstring: ARR lives
    only in the CRM export)."""
    arr_by_crm_id = load_account_arr_via_connector(full_graph.connector)
    return account_arr_by_note_slug(full_graph.conn, arr_by_crm_id)


def lint_error_identity(finding) -> tuple:
    """A stable identity for one `LintFinding` (check + whichever of
    note_slug/claim_id/source_slug it set), ignoring `message` -- some
    checks' messages embed counts or dates that can shift run to run
    without the underlying problem being a *new* one. Used to diff
    `FullGraph.baseline_lint_errors` against a later `run_lint` call."""
    return (finding.check, finding.note_slug, finding.claim_id, finding.source_slug)
