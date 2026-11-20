"""Tests for `portolan.web` (build-contracts.md "Web and MCP", wave C3).

One graph, built once per test session by running the real pipeline
(`portolan.pipeline.run`) with the fake extractor over `corpus/out` --
"a graph from the fake pipeline run in a session fixture" (build
instructions "Tests"). Individual tests then query that graph for a
suitable note/claim/source to exercise a route against, rather than
hardcoding ids the pipeline's own internals could shift.

Run with `DATABASE_URL=postgresql://localhost/portolan_test_web` (own
test database per build-contracts.md's "Own test database" rule).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree

import psycopg
import pytest
from fastapi.testclient import TestClient
from itsdangerous import URLSafeTimedSerializer

from portolan import db as db_module
from portolan.config import Settings
from portolan.connectors.file_connector import FileConnector
from portolan.fake_extractor import FakePlacements, build_fake_llm_client
from portolan.llm.fake import FakeLLMClient
from portolan.notes.render import CITATION_RE
from portolan.pipeline import run as pipeline_run
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.web.app import create_app
from portolan.web.auth import SESSION_COOKIE_NAME, SESSION_COOKIE_SALT
from portolan.web.deps import get_conn, get_embedder_dep, get_llm_dep, get_store_dep

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_OUT = REPO_ROOT / "corpus" / "out"
ORG_KEY = "test-org-key-for-web-tests"
# Fix wave 2026-09-12 (review finding F1): PORTOLAN_SESSION_SECRET is now
# required -- Settings.session_secret, passed explicitly into every
# Settings(...) this module builds, rather than via
# monkeypatch.setenv(...) after the app (and its one-time
# _require_auth_settings check) already exists. This is a real secret
# value only in the sense that it is *a* secret string; it is a test
# fixture, not anything reused outside this file.
SESSION_SECRET = "test-session-secret-for-web-tests-not-a-real-secret"


# --------------------------------------------------------------------
# Session-scoped graph
# --------------------------------------------------------------------


@pytest.fixture(scope="session")
def store_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("web-store")


@pytest.fixture(scope="session")
def demo_conn(db_url: str, store_root: Path):
    """A freshly-migrated schema, then one real `pipeline.run` over
    `corpus/out` with the fake extractor -- built once for every test in
    this module. Skips (matching `tests/conftest.py`'s `db` fixture) if
    Postgres/pgvector aren't reachable.
    """
    try:
        conn = db_module.connect(db_url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unreachable at {db_url!r}: {exc}")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA public CASCADE")
            cur.execute("CREATE SCHEMA public")
        conn.commit()
        db_module.apply_migrations(conn)
    except psycopg.errors.UndefinedFile as exc:  # pragma: no cover - defensive
        conn.close()
        pytest.skip(f"pgvector extension file not found on this Postgres server: {exc}")
        return
    except psycopg.errors.FeatureNotSupported as exc:
        conn.close()
        pytest.skip(f"pgvector extension not installed on this Postgres server: {exc}")
        return
    conn.commit()
    conn.autocommit = True

    settings = Settings(
        database_url=db_url,
        tenant="test",
        store_root=store_root,
        llm_provider="fake",
        embed_provider="fake",
        org_key=ORG_KEY,
        session_secret=SESSION_SECRET,
    )
    connector = FileConnector(CORPUS_OUT)
    llm = build_fake_llm_client(FakePlacements(CORPUS_OUT), tenant=settings.tenant)
    embedder = FakeEmbedder()
    store = LocalStore(store_root)
    registry = Registry.load(conn)

    pipeline_run(
        conn,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
    )

    yield conn
    conn.close()


# --------------------------------------------------------------------
# App / client
# --------------------------------------------------------------------


def _settings(db_url: str, store_root: Path, **overrides) -> Settings:
    defaults = dict(
        database_url=db_url,
        tenant="test",
        store_root=store_root,
        llm_provider="fake",
        embed_provider="fake",
        org_key=ORG_KEY,
        session_secret=SESSION_SECRET,
        # TestClient's ASGI transport is always plain http://testserver,
        # never https -- a Secure-flagged cookie (Settings' own default,
        # correctly, for anything real) never round-trips through it,
        # confirmed directly against httpx/starlette: a client-jar cookie
        # set with secure=True comes back as None on the very next
        # request. This is the "http://127.0.0.1 dev" case the flag
        # exists for, so it's off by default for every test in this
        # module; the one test that checks Secure IS present by default
        # (test_session_cookie_is_secure_httponly_samesite_lax) builds
        # its own separate app instead of using this default.
        cookie_secure=False,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture()
def client(demo_conn: psycopg.Connection, db_url: str, store_root: Path):
    """A `TestClient` wired to the session graph: `get_conn` is overridden
    to hand out the *same* connection every test in this module shares
    (so setup SQL and route reads/writes always see each other), and the
    embedder/store are overridden to fakes/the shared local store. `llm`
    is deliberately left at `create_app`'s own default (`FakeLLMClient([])`,
    zero canned responses) -- tests that call `/chat` override
    `get_llm_dep` themselves with the canned answer they want.

    `org_key`/`session_secret` are passed straight into `Settings(...)`
    (fix wave 2026-09-12, review finding F1) rather than set via
    `monkeypatch.setenv(...)` after the fact: `create_app` now reads both
    once, at app-construction time, and refuses to build at all if
    either is missing -- there is no later point on this object a test
    could still change them.
    """
    settings = _settings(db_url, store_root)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_cookies(client: TestClient):
    r = client.post(
        "/login", data={"org_key": ORG_KEY, "editor": "Test Editor"}, follow_redirects=False
    )
    assert r.status_code == 303
    return r.cookies


# --------------------------------------------------------------------
# Small query helpers (fixtures) for a suitable note/claim/source to test against
# --------------------------------------------------------------------


def _one(conn: psycopg.Connection, sql: str, params: tuple = ()) -> tuple | None:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def _note_version(conn: psycopg.Connection, slug: str) -> int:
    """The highest `note_versions.version` recorded for `slug`, or 0 if
    it has never been edited -- `note_versions` only ever holds *prior*
    versions (a snapshot taken before a change), never the current one."""
    return _one(
        conn, "SELECT coalesce(max(version), 0) FROM note_versions WHERE slug = %s", (slug,)
    )[0]


@pytest.fixture()
def any_claim(demo_conn: psycopg.Connection) -> tuple[str, int, str, str]:
    """`(note_slug, claim_id, source_slug, locator)` for some real claim."""
    row = _one(
        demo_conn,
        "SELECT note_slug, id, source_slug, locator FROM claims ORDER BY id LIMIT 1",
    )
    assert row is not None, "demo graph has no claims"
    return row


@pytest.fixture()
def note_with_links_both_directions(demo_conn: psycopg.Connection) -> tuple[str, str, str]:
    """`(slug, outgoing_slug, incoming_slug)` -- a note that has at least
    one outgoing body/parent/supersession link (to `outgoing_slug`) AND at
    least one incoming one (from `incoming_slug`), for `/graph/note/
    {slug}.json`'s "links both directions" test (restyle wave 2026-09-15
    v2)."""
    row = _one(
        demo_conn,
        """
        SELECT out_l.from_slug, out_l.to_slug, in_l.from_slug
        FROM links out_l
        JOIN links in_l ON in_l.to_slug = out_l.from_slug
        JOIN notes n_from ON n_from.slug = out_l.from_slug
        JOIN notes n_to ON n_to.slug = out_l.to_slug
        JOIN notes n_in ON n_in.slug = in_l.from_slug
        WHERE out_l.kind = ANY(%s) AND in_l.kind = ANY(%s)
          AND out_l.to_slug != in_l.from_slug
        LIMIT 1
        """,
        (["body", "parent", "supersession"], ["body", "parent", "supersession"]),
    )
    assert row is not None, "demo graph has no note with both an outgoing and an incoming link"
    return row


@pytest.fixture()
def call_source(demo_conn: psycopg.Connection) -> tuple[str, str]:
    """`(source_slug, locator)` for a claim whose source is a call."""
    row = _one(
        demo_conn,
        """
        SELECT c.source_slug, c.locator FROM claims c
        JOIN sources s ON s.slug = c.source_slug
        WHERE s.source_type = 'call'
        ORDER BY c.id LIMIT 1
        """,
    )
    assert row is not None, "demo graph has no call-sourced claim"
    return row


@pytest.fixture()
def email_source(demo_conn: psycopg.Connection) -> tuple[str, str]:
    row = _one(
        demo_conn,
        """
        SELECT c.source_slug, c.locator FROM claims c
        JOIN sources s ON s.slug = c.source_slug
        WHERE s.source_type = 'email'
        ORDER BY c.id LIMIT 1
        """,
    )
    assert row is not None, "demo graph has no email-sourced claim"
    return row


@pytest.fixture()
def ticket_source(demo_conn: psycopg.Connection) -> tuple[str, str]:
    row = _one(
        demo_conn,
        """
        SELECT c.source_slug, c.locator FROM claims c
        JOIN sources s ON s.slug = c.source_slug
        WHERE s.source_type = 'ticket'
        ORDER BY c.id LIMIT 1
        """,
    )
    assert row is not None, "demo graph has no ticket-sourced claim"
    return row


# --------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------


def test_root_redirects_without_session(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login"


def test_login_page_ok(client: TestClient) -> None:
    assert client.get("/login").status_code == 200


def test_login_wrong_key_rejected(client: TestClient) -> None:
    r = client.post("/login", data={"org_key": "definitely-wrong", "editor": "Ada"})
    assert r.status_code == 401


def test_login_works_and_session_persists(client: TestClient, auth_cookies) -> None:
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    assert "Test Editor" in r.text


# --------------------------------------------------------------------
# Bug hunt 2026-09-18: `POST /login` throttling (finding 1). `test_login_
# wrong_key_rejected` above proved 20 unlimited guesses used to be
# accepted; these prove the fix -- a small per-IP limit trips after N
# failures, the next attempt (even with the *right* key) gets 429 with a
# Retry-After, and the failure itself is logged with the IP but never the
# submitted key.
# --------------------------------------------------------------------


def test_login_throttles_after_repeated_failures_from_one_ip(
    demo_conn: psycopg.Connection, db_url: str, store_root: Path
) -> None:
    settings = _settings(
        db_url,
        store_root,
        login_rate_limit_per_ip=3,
        login_rate_limit_window_seconds=60,
        login_rate_limit_backoff_base_seconds=60,
    )
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as c:
        for _ in range(3):
            r = c.post("/login", data={"org_key": "wrong-key", "editor": "Mallory"})
            assert r.status_code == 401

        # The 4th attempt is blocked before the key is even checked --
        # the *real* key doesn't get through either.
        blocked = c.post("/login", data={"org_key": ORG_KEY, "editor": "Mallory"})
        assert blocked.status_code == 429
        assert "Too many attempts" in blocked.text
        retry_after = int(blocked.headers["Retry-After"])
        assert retry_after > 0


def test_login_failure_is_logged_with_ip_and_never_the_submitted_key(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    secret_guess = "definitely-wrong-super-secret-guess-value"
    with caplog.at_level(logging.WARNING, logger="portolan.web.routes.login"):
        r = client.post("/login", data={"org_key": secret_guess, "editor": "Mallory"})
    assert r.status_code == 401
    messages = [rec.message for rec in caplog.records if rec.name == "portolan.web.routes.login"]
    assert any("login failed" in m and "ip=" in m for m in messages)
    assert not any(secret_guess in m for m in messages)


def test_login_rate_limit_ignores_x_forwarded_for_unless_trusted(
    demo_conn: psycopg.Connection, db_url: str, store_root: Path
) -> None:
    """`trust_forwarded_for` defaults to False (Railway-behind-a-proxy
    note in `config.py`): a caller can't pick its own rate-limit bucket
    by sending a fake `X-Forwarded-For` unless the deployment explicitly
    says to trust it."""
    settings = _settings(db_url, store_root, login_rate_limit_per_ip=2, trust_forwarded_for=False)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as c:
        for i in range(2):
            r = c.post(
                "/login",
                data={"org_key": "wrong", "editor": "M"},
                headers={"X-Forwarded-For": f"1.2.3.{i}"},
            )
            assert r.status_code == 401
        # A third, differently-"forwarded" IP still hits the same bucket
        # -- the real TestClient connection is what's actually keyed on.
        blocked = c.post(
            "/login",
            data={"org_key": "wrong", "editor": "M"},
            headers={"X-Forwarded-For": "9.9.9.9"},
        )
        assert blocked.status_code == 429


def test_login_rate_limit_keys_on_x_forwarded_for_when_trusted(
    demo_conn: psycopg.Connection, db_url: str, store_root: Path
) -> None:
    settings = _settings(db_url, store_root, login_rate_limit_per_ip=2, trust_forwarded_for=True)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as c:
        for _ in range(2):
            r = c.post(
                "/login",
                data={"org_key": "wrong", "editor": "M"},
                headers={"X-Forwarded-For": "5.5.5.5"},
            )
            assert r.status_code == 401
        blocked = c.post(
            "/login",
            data={"org_key": "wrong", "editor": "M"},
            headers={"X-Forwarded-For": "5.5.5.5"},
        )
        assert blocked.status_code == 429
        # A genuinely different forwarded IP is a separate bucket.
        other_ip = c.post(
            "/login",
            data={"org_key": ORG_KEY, "editor": "Someone Else"},
            headers={"X-Forwarded-For": "6.6.6.6"},
            follow_redirects=False,
        )
        assert other_ip.status_code == 303


# --------------------------------------------------------------------
# How it works (fix wave 2026-09-13): the self-contained explainer at
# 01-spec/how-portolan-works.html, served public -- no session required.
# --------------------------------------------------------------------


def test_how_it_works_is_public_and_contains_wordmark_and_audio(client: TestClient) -> None:
    client.cookies.clear()
    r = client.get("/how-it-works")
    assert r.status_code == 200
    assert "wordmark" in r.text
    assert "<audio" in r.text


def test_how_it_works_has_a_home_link_back_to_the_app(client: TestClient) -> None:
    """This page is served standalone, publicly, no session, no nav -- a
    reader who lands here (a shared link, a search engine) had no way
    back into the app at all. `.home-link` (`01-spec/how-portolan-works.html`)
    fixes that -- styled like the page's own `.theme-btn`, pointing at `/`."""
    client.cookies.clear()
    r = client.get("/how-it-works")
    assert r.status_code == 200
    assert 'class="home-link" href="/"' in r.text


def test_how_it_works_is_cheap_on_repeat_requests(client: TestClient) -> None:
    """Review finding 6: this was a 2.85MB-per-request amplifier with no
    caching at all -- a fresh `200` full body on every single `GET`, even
    one immediately repeating the etag it had just issued. Fix: a real
    `Cache-Control`, and a matching `If-None-Match` gets a bare `304`."""
    client.cookies.clear()
    first = client.get("/how-it-works")
    assert first.status_code == 200
    assert first.headers["cache-control"] == "public, max-age=3600"
    etag = first.headers.get("etag")
    assert etag

    second = client.get("/how-it-works", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.content == b""
    assert second.headers.get("cache-control") == "public, max-age=3600"


def test_how_it_works_route_never_grows_a_path_parameter(client: TestClient) -> None:
    """Regression guard (review Test gaps): this route serves exactly one
    file -- a path parameter here would turn the public, unauthenticated
    route into a traversal surface."""
    schema = client.app.openapi()
    how_it_works_paths = [p for p in schema["paths"] if p.startswith("/how-it-works")]
    assert how_it_works_paths == ["/how-it-works"]
    parameters = schema["paths"]["/how-it-works"]["get"].get("parameters", [])
    assert parameters == []


def test_how_it_works_content_is_read_once_at_import_not_per_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding 6: "the file is read once at startup or cached, not
    re-read per request." `portolan.web.routes.how_it_works._CONTENT` is
    populated at import time; patching `Path.read_bytes` to explode
    proves the route never reads the file again for a live request."""
    from pathlib import Path

    from portolan.web.routes import how_it_works as how_it_works_module

    assert how_it_works_module._CONTENT is not None, "must already be cached from import time"

    def _boom(self, *args, **kwargs):
        raise AssertionError("the route must not re-read the file per request")

    monkeypatch.setattr(Path, "read_bytes", _boom)
    client.cookies.clear()
    r = client.get("/how-it-works")
    assert r.status_code == 200


def test_how_it_works_linked_from_login_page(client: TestClient) -> None:
    r = client.get("/login")
    assert 'href="/how-it-works"' in r.text


def test_how_it_works_linked_from_nav_when_logged_in(client: TestClient, auth_cookies) -> None:
    r = client.get("/", cookies=auth_cookies)
    assert 'href="/how-it-works"' in r.text


# --------------------------------------------------------------------
# Page shell (UI wave 1, item 1): the nav renders on every page, a
# footer with the working name and app version, dark mode's toggle and
# pre-paint script, and the brand-CSS-escaping fix (coordinator directive
# following the concurrent UX evaluation, 01-spec/reviews/
# ux-eval-2026-09-12.md punch item 1).
# --------------------------------------------------------------------

_SHELL_SWEEP_PAGES = ["/", "/atlas", "/graph", "/reports", "/coverage", "/review"]


@pytest.mark.parametrize("path", _SHELL_SWEEP_PAGES)
def test_shell_nav_present_on_every_page(client: TestClient, auth_cookies, path: str) -> None:
    r = client.get(path, cookies=auth_cookies)
    assert r.status_code == 200
    for nav_href in ('href="/"', "/atlas", "/graph", "/reports", "/coverage", "/review"):
        assert nav_href in r.text, f"{path} is missing a nav link to {nav_href}"
    assert "Test Editor" in r.text, f"{path} is missing the signed-in name"
    assert 'action="/logout"' in r.text, f"{path} is missing the sign-out control"
    assert "Working name" in r.text, f"{path} is missing the footer"
    assert 'id="theme-toggle"' in r.text, f"{path} is missing the theme toggle"


def test_shell_footer_shows_the_app_version(client: TestClient, auth_cookies) -> None:
    import re as re_module

    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    assert re_module.search(r'class="mono">v[\w.]+</span>', r.text), (
        "footer is missing a mono app-version stamp"
    )
    assert "design as of 2026-09-12" in r.text


def test_shell_has_no_nav_on_the_logged_out_login_page_but_still_has_a_toggle(
    client: TestClient,
) -> None:
    r = client.get("/login")
    assert r.status_code == 200
    assert "/search" not in r.text  # no nav for a logged-out visitor
    assert 'id="theme-toggle"' in r.text  # dark mode works pre-login too


def test_dark_mode_pre_paint_script_and_css_are_present(client: TestClient, auth_cookies) -> None:
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    assert "localStorage.getItem('portolan-theme')" in r.text


def test_mobile_nav_closes_on_escape(client: TestClient, auth_cookies) -> None:
    """Review finding 22 (grouped nits): the UX eval's nav requirement
    asked for Escape to close the mobile nav; only a second tap on the
    hamburger did."""
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    assert "e.key === 'Escape'" in r.text
    assert "nav.classList.contains('open')" in r.text
    assert "data-theme" in r.text
    assert ':root[data-theme="dark"]' in r.text
    assert "prefers-color-scheme: dark" in r.text


def test_brand_css_is_not_html_escaped_in_the_style_block(client: TestClient, auth_cookies) -> None:
    """Critical fix, coordinator directive following the concurrent UX
    evaluation: `{{ brand_tokens_css }}` used to go through Jinja's
    autoescaper with no `|safe`, turning every `"` into `&#34;` --
    `--pl-display: "Avenir Next", ...` became an invalid custom-property
    value (every font fell back to Times) and `:root[data-theme="dark"]`
    became an invalid, always-false attribute selector (dark mode did
    nothing). Asserts the rendered page's <style> block contains real
    quote characters and no HTML-entity-encoded ones.
    """
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    style_start = r.text.index("<style>")
    style_end = r.text.index("</style>", style_start)
    style_block = r.text[style_start:style_end]
    assert "&#34;" not in style_block
    assert '--pl-display: "Avenir Next"' in style_block
    assert ':root[data-theme="dark"]' in style_block


def test_global_links_are_not_all_accent_and_citation_chip_keeps_it(
    client: TestClient, auth_cookies
) -> None:
    """Design-system fix alongside the CSS-escaping one: the accent is
    reserved for the active nav item, one primary button, the citation
    chip, and the highlighted locator band -- not every hyperlink on the
    page (01-spec/reviews/ux-eval-2026-09-12.md punch item 3)."""
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    style_start = r.text.index("<style>")
    style_end = r.text.index("</style>", style_start)
    style_block = r.text[style_start:style_end]
    assert "a { color: var(--pl-ink)" in style_block
    assert "a:hover, a:focus-visible { color: var(--pl-accent)" in style_block
    assert (
        "border: 1px solid var(--pl-accent); color: var(--pl-accent); text-decoration: none;"
        in (style_block.replace("\n", " "))
    )


# --------------------------------------------------------------------
# F1: no auth fail-open -- create_app refuses to start, and a cookie
# forged with any guessable secret is rejected.
# --------------------------------------------------------------------


def test_create_app_refuses_to_start_without_session_secret(db_url: str, store_root: Path) -> None:
    settings = _settings(db_url, store_root, session_secret=None)
    with pytest.raises(RuntimeError, match="PORTOLAN_SESSION_SECRET"):
        create_app(settings)


def test_create_app_refuses_to_start_without_org_key(db_url: str, store_root: Path) -> None:
    settings = _settings(db_url, store_root, org_key=None)
    with pytest.raises(RuntimeError, match="PORTOLAN_ORG_KEY"):
        create_app(settings)


def test_build_templates_raises_when_brand_tokens_css_is_missing_or_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review finding 21: a deployment whose `04-brand/` didn't ship used
    to render every page completely unstyled with no error anywhere --
    the exact symptom UX punch item 1 was about. `build_templates()` now
    fails loudly, at app-startup time, instead."""
    from portolan.web import templating as templating_module

    monkeypatch.setattr(templating_module, "_read_text", lambda path: "")
    with pytest.raises(RuntimeError, match=r"brand-tokens\.css"):
        templating_module.build_templates()


def test_forged_cookie_with_a_guessable_secret_is_rejected(client: TestClient) -> None:
    """Review finding F1, proven end to end against a real running app:
    `portolan/web/auth.py` used to fall back to the literal
    `"portolan-dev-insecure-secret-set-PORTOLAN_ORG_KEY"` when neither
    secret was configured, and a cookie forged with that exact string
    returned full note bodies on a real deployment. That literal fallback
    is gone; this proves a cookie signed with *any* string other than
    this app's real `session_secret` -- guessable literal or otherwise --
    is rejected, not merely the one specific string the review used."""
    for guess in (
        "portolan-dev-insecure-secret-set-PORTOLAN_ORG_KEY",
        ORG_KEY,  # a plausible guess: reuse the org key as the secret
        "",
        "changeme",
    ):
        forged = URLSafeTimedSerializer(guess, salt=SESSION_COOKIE_SALT).dumps(
            {"editor": "attacker"}
        )
        r = client.get("/", cookies={SESSION_COOKIE_NAME: forged}, follow_redirects=False)
        assert r.status_code == 302, f"a cookie forged with {guess!r} was accepted"


def test_forged_cookie_with_the_real_secret_but_wrong_signature_is_rejected(
    client: TestClient,
) -> None:
    forged = URLSafeTimedSerializer(SESSION_SECRET, salt=SESSION_COOKIE_SALT).dumps(
        {"editor": "attacker"}
    )
    # Tamper the SECOND-to-last character, not the last. The cookie's
    # trailing base64 signature has no `=` padding, so its very last
    # character only carries the leftover bits of the final byte (a
    # 256-bit HMAC-SHA256 signature base64-encodes to 43 chars: 42 full
    # 6-bit groups plus one 4-bit-meaningful/2-bit-unused tail char). A
    # 1-in-4 swap of that tail char (measured empirically at ~19% over
    # 200k trials, `/tmp/probe_sig.py` this fix wave) lands on a
    # different base64 CHARACTER that decodes to the SAME leftover bits,
    # so the signature still verifies and this test flakes green on a
    # cookie that was, in fact, tampered with. Every character before
    # the last sits in a full, unambiguous 6-bit group, so tampering
    # there always changes the decoded signature bytes -- confirmed
    # deterministic (0/200k) the same way.
    idx = -2
    tampered_char = "A" if forged[idx] != "A" else "B"
    tampered = forged[:idx] + tampered_char + forged[idx + 1 :]
    r = client.get("/", cookies={SESSION_COOKIE_NAME: tampered}, follow_redirects=False)
    assert r.status_code == 302


# --------------------------------------------------------------------
# F2: session cookie flags and expiry
# --------------------------------------------------------------------


def test_session_cookie_is_secure_httponly_samesite_lax(
    demo_conn: psycopg.Connection, db_url: str, store_root: Path
) -> None:
    # A dedicated app, not the shared `client` fixture: that fixture
    # turns cookie_secure off so its cookies round-trip through
    # TestClient's plain-http transport (see _settings' docstring). This
    # test only ever inspects the Set-Cookie header of the login
    # response itself -- it never needs the cookie to be sent back -- so
    # it can use Settings.cookie_secure's real default (True) and still
    # pass.
    settings = _settings(db_url, store_root, cookie_secure=True)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as c:
        r = c.post(
            "/login",
            data={"org_key": ORG_KEY, "editor": "Cookie Checker"},
            follow_redirects=False,
        )
    assert r.status_code == 303
    set_cookie = r.headers.get("set-cookie", "")
    assert SESSION_COOKIE_NAME in set_cookie
    assert "HttpOnly" in set_cookie
    assert "samesite=lax" in set_cookie.lower()
    assert "Secure" in set_cookie


def test_session_cookie_secure_can_be_turned_off_for_dev(
    demo_conn: psycopg.Connection, db_url: str, store_root: Path
) -> None:
    settings = _settings(db_url, store_root, cookie_secure=False)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as c:
        r = c.post("/login", data={"org_key": ORG_KEY, "editor": "Dev"}, follow_redirects=False)
    assert "Secure" not in r.headers.get("set-cookie", "")


def test_session_cookie_expires_after_eight_hours_on_read(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from portolan.web.auth import SESSION_MAX_AGE_SECONDS

    assert SESSION_MAX_AGE_SECONDS == 60 * 60 * 8

    cookie = URLSafeTimedSerializer(SESSION_SECRET, salt=SESSION_COOKIE_SALT).dumps(
        {"editor": "Old Session"}
    )
    # A cookie that's still fresh is accepted...
    r = client.get("/", cookies={SESSION_COOKIE_NAME: cookie}, follow_redirects=False)
    assert r.status_code == 200

    # ...but itsdangerous enforces max_age on *read*, using the payload's
    # own signed timestamp -- simulate an 8-hour-old cookie by signing it
    # under a backdated clock rather than sleeping in a test.
    import time as time_module

    import itsdangerous.timed as timed_module

    real_now = time_module.time()
    monkeypatch.setattr(
        timed_module.time, "time", lambda: real_now - (SESSION_MAX_AGE_SECONDS + 60)
    )
    old_cookie = URLSafeTimedSerializer(SESSION_SECRET, salt=SESSION_COOKIE_SALT).dumps(
        {"editor": "Old Session"}
    )
    monkeypatch.undo()

    r = client.get("/", cookies={SESSION_COOKIE_NAME: old_cookie}, follow_redirects=False)
    assert r.status_code == 302, "an 8-hour-old session cookie should be rejected on read"


# --------------------------------------------------------------------
# F3/F9: unauthenticated surfaces closed -- /docs, /redoc, /openapi.json
# off by default, gated behind a session when enabled; every registered
# route re-checked by an automated sweep.
# --------------------------------------------------------------------


def test_docs_are_off_by_default(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 404, f"{path} should not exist when docs are disabled"


def test_docs_require_a_session_when_enabled(
    demo_conn: psycopg.Connection, db_url: str, store_root: Path
) -> None:
    settings = _settings(db_url, store_root, enable_api_docs=True)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as c:
        for path in ("/docs", "/redoc", "/openapi.json"):
            r = c.get(path, follow_redirects=False)
            assert r.status_code == 302, f"{path} should require a session"

        login = c.post(
            "/login", data={"org_key": ORG_KEY, "editor": "Docs Reader"}, follow_redirects=False
        )
        cookies = login.cookies
        for path in ("/docs", "/redoc", "/openapi.json"):
            r = c.get(path, cookies=cookies)
            assert r.status_code == 200, f"{path} should be reachable with a session"


_ROUTE_SWEEP_SKIP = {
    "/login",
    "/logout",
    # Fix wave 2026-09-13, build instructions: the explainer is meant to be
    # public -- "Serve ... public (no login)" -- reachable by anyone before
    # they've ever logged in, unlike everything else on this route sweep.
    "/how-it-works",
}


def test_every_registered_route_rejects_a_request_with_no_session(client: TestClient) -> None:
    """Review finding F9: "every route re-checked by an automated sweep
    test that requests each registered route without a session and
    asserts 302/401." Reads the app's own OpenAPI schema (`app.openapi()`)
    for the path/method table rather than hand-listing paths or walking
    `app.routes` directly -- the installed FastAPI's `.routes` wraps
    `include_router`-mounted routes in an internal `_IncludedRouter`
    object with no `.path`/`.methods` of its own, so the schema is the
    stable, public way to enumerate what's actually registered. A route
    added later is covered automatically.
    """
    import re as re_module

    client.cookies.clear()
    schema = client.app.openapi()
    seen_any = False
    for path, operations in schema["paths"].items():
        if path in _ROUTE_SWEEP_SKIP:
            continue
        concrete_path = re_module.sub(r"\{[^}]+\}", "x", path)
        for method in operations:
            if method.upper() in ("HEAD", "OPTIONS"):
                continue
            seen_any = True
            r = client.request(method.upper(), concrete_path, follow_redirects=False)
            assert r.status_code in (302, 401), (
                f"{method.upper()} {path} returned {r.status_code} with no session, "
                "expected 302 or 401"
            )
    assert seen_any, "route sweep found nothing to check -- the OpenAPI schema had no paths"


def test_mcp_route_rejects_a_valid_web_session_cookie_with_no_bearer(
    demo_conn: psycopg.Connection, db_url: str, store_root: Path
) -> None:
    """Security review 2026-09-19, S5: `/mcp` is registered with
    `include_in_schema=False` (`web/app.py`'s `_mount_mcp`), so the
    OpenAPI-schema-driven sweep above structurally cannot see it -- the
    one route serving note bodies and claim quotes over the internet is
    invisible to the app's own security sweep, and `/mcp` carries no
    `Depends(require_session)` for that sweep to even check if it could.
    Checked directly instead: a real, freshly logged-in, validly-signed
    session cookie -- the same kind `require_session` accepts everywhere
    else in this app -- sent with no `Authorization` header at all, must
    still get the bearer gate's 401, not a free pass through the cookie
    layer. A regression that let a cookie session substitute for the MCP
    bearer would be invisible to every other test in this suite.
    """
    settings = _settings(db_url, store_root, mcp_org_key="a" * 32)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)

    with TestClient(app) as c:
        login = c.post(
            "/login",
            data={"org_key": ORG_KEY, "editor": "MCP sweep"},
            follow_redirects=False,
        )
        assert login.status_code == 303
        cookies = login.cookies
        assert cookies.get(SESSION_COOKIE_NAME), "expected a real session cookie from login"

        r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, cookies=cookies)

    assert r.status_code == 401


# --------------------------------------------------------------------
# Chat: sessions, streaming, citation validation (UI wave 1)
# --------------------------------------------------------------------


def _parse_sse_events(text: str) -> list[tuple[str, dict]]:
    """`[(event_name, payload_dict), ...]` out of a raw SSE response body
    (`routes/chat.py`'s `event: <name>\\ndata: <json>\\n\\n` framing)."""
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_line = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_line = line[len("data:") :].strip()
        if data_line:
            events.append((event_name, json.loads(data_line)))
    return events


def test_chat_stream_requires_session(client: TestClient) -> None:
    r = client.post("/chat/new/stream", data={"question": "anything"})
    assert r.status_code == 401


def test_chat_stream_new_conversation_drops_sentence_citing_unknown_slug_and_persists_rows(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Streaming replaces the old `/chat` fragment endpoint, but must not
    weaken `answer.py`'s citation validation (build instructions: "keep
    answer.py's citation validation exactly as is; the streaming wrapper
    must not weaken it") -- same drop-the-unvalidated-sentence behaviour
    as before, proven against the real SSE body, plus the new persistence
    contract: a "new" stream creates the conversation, and writes exactly
    one user message, one assistant message, and one usage_events row.
    """
    from portolan.search.embed import FakeEmbedder
    from portolan.search.query import search as run_search

    question = "What happened with the account, streaming edition?"
    result = run_search(demo_conn, FakeEmbedder(), question, k=8, expand=1)
    assert result.hits, "expected the demo graph to return at least one hit"

    # Pick the first HIT that actually has a claim to cite, rather than
    # assuming `result.hits[0]` always does: some retrievable notes (an
    # account/product/competitor whose only fact was an "other"/entity-
    # only fact, e.g. `ashgrove-health`'s ARR-from-CRM fact) legitimately
    # carry zero claims of their own once merge routes that fact through
    # a wrapping signal note instead (portolan.merge.facts.
    # _ensure_signal_note's docstring, e2e-run follow-up wave item 5) --
    # this test only needs SOME real, retrieved claim to ground a
    # citation in, not specifically the top-ranked note's own.
    retrieved_slug = None
    claim_row = None
    for hit in result.hits:
        row = _one(
            demo_conn,
            "SELECT id, source_slug, locator FROM claims WHERE note_slug = %s ORDER BY id LIMIT 1",
            (hit.note_slug,),
        )
        if row is not None:
            retrieved_slug = hit.note_slug
            claim_row = row
            break
    assert claim_row is not None, (
        f"none of the top {len(result.hits)} search hits "
        f"({[h.note_slug for h in result.hits]}) have any claims to cite"
    )
    claim_id, source_slug, locator = claim_row

    canned = {
        "sentences": [
            {
                "text": "This is a real, grounded sentence.",
                "note_slugs": [retrieved_slug],
                "claim_ids": [claim_id],
            },
            {
                "text": "This sentence cites nothing real and must be dropped.",
                "note_slugs": ["not-a-real-note-slug"],
                "claim_ids": [999999999],
            },
        ]
    }
    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([canned])

    usage_before = _one(demo_conn, "SELECT count(*) FROM usage_events WHERE surface = 'chat'")[0]

    r = client.post("/chat/new/stream", data={"question": question}, cookies=auth_cookies)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "This is a real, grounded sentence." in r.text
    assert "This sentence cites nothing real" not in r.text
    assert f"/notes/{retrieved_slug}#claim-{claim_id}" in r.text
    assert f"/originals/{source_slug}?at={quote(locator, safe='/')}" in r.text

    events = _parse_sse_events(r.text)
    sentence_events = [p for name, p in events if name == "sentence"]
    done_events = [p for name, p in events if name == "done"]
    assert len(sentence_events) == 1, "the dropped sentence must not be streamed either"
    assert len(done_events) == 1
    # Fix wave 2026-09-13: the "done" event now names the model that
    # actually answered (chat.html's live "Answering with <model>"
    # footnote reads this). This fixture's settings never set CHAT_MODEL,
    # so it must be whatever `strong_model` resolves to (picked up from
    # this checkout's own .env -- see the dedicated
    # test_chat_uses_chat_model_setting_falling_back_to_strong_model for
    # the exact CHAT_MODEL-set-vs-unset behaviour with both pinned) --
    # never the FakeLLMClient's own default `model_key` ("nova-lite"),
    # which is what a missing `model=` argument would have produced.
    assert done_events[0]["model"] not in ("", "nova-lite")
    conversation_id = done_events[0]["conversation_id"]

    usage_after = _one(demo_conn, "SELECT count(*) FROM usage_events WHERE surface = 'chat'")[0]
    assert usage_after == usage_before + 1

    with demo_conn.cursor() as cur:
        cur.execute(
            "SELECT role, content_md FROM messages WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        )
        rows = cur.fetchall()
    assert [row[0] for row in rows] == ["user", "assistant"]
    assert rows[0][1] == question
    assert "This is a real, grounded sentence." in rows[1][1]
    assert "This sentence cites nothing real" not in rows[1][1]

    owner = _one(
        demo_conn, "SELECT owner, title FROM conversations WHERE id = %s", (conversation_id,)
    )
    assert owner[0] == "Test Editor"
    # Owner verdict 2026-09-18: the auto-title is now the question's first
    # ~6 words (portolan.web.conversations.title_from_question), not a
    # 60-character slice -- see the dedicated test_title_from_question_*
    # tests for the word-count rule itself.
    from portolan.web.conversations import title_from_question

    assert owner[1] == title_from_question(question)

    # The same model name, server-rendered as the "Answering with <model>"
    # footnote (chat.html) on a plain page reload of that conversation --
    # reads `messages.usage.model`, the same field the live SSE "done"
    # event carried above, so a reload never disagrees with what streamed.
    reload_page = client.get(f"/c/{conversation_id}", cookies=auth_cookies)
    assert reload_page.status_code == 200
    assert "Answering with " + done_events[0]["model"] in reload_page.text

    client.app.dependency_overrides.pop(get_llm_dep, None)


def test_chat_footnote_says_how_many_sentences_were_withheld_for_no_citation(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """A sentence dropped for citing nothing valid is counted in the stream's
    `done` event and in the stored usage, so both the live footnote and a
    reload say it was withheld instead of the answer silently shrinking."""
    from portolan.search.embed import FakeEmbedder
    from portolan.search.query import search as run_search

    question = "What happened with the account, withheld edition?"
    hits = run_search(demo_conn, FakeEmbedder(), question, k=8, expand=1).hits
    claim_row = None
    for hit in hits:
        claim_row = _one(
            demo_conn,
            "SELECT id, note_slug FROM claims WHERE note_slug = %s ORDER BY id LIMIT 1",
            (hit.note_slug,),
        )
        if claim_row is not None:
            break
    assert claim_row is not None, "expected a retrieved note with a claim to cite"
    claim_id, note_slug = claim_row

    canned = {
        "sentences": [
            {"text": "A grounded sentence.", "note_slugs": [note_slug], "claim_ids": [claim_id]},
            {"text": "An uncited sentence.", "note_slugs": [], "claim_ids": []},
            {"text": "Another uncited one.", "note_slugs": ["no-such-note"], "claim_ids": []},
        ]
    }
    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([canned])
    try:
        r = client.post("/chat/new/stream", data={"question": question}, cookies=auth_cookies)
        assert r.status_code == 200
        done = [p for name, p in _parse_sse_events(r.text) if name == "done"]
        assert done[0]["uncited_dropped"] == 2
        assert done[0]["numerals_dropped"] == 0
        assert "An uncited sentence." not in r.text

        reload_page = client.get(f"/c/{done[0]['conversation_id']}", cookies=auth_cookies)
        assert reload_page.status_code == 200
        assert "2 sentences withheld: no citation" in reload_page.text
    finally:
        client.app.dependency_overrides.pop(get_llm_dep, None)


def test_chat_citation_chip_shows_the_notes_title_not_its_slug_with_date_only_on_hover(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Demo-polish fix (app checklist 2026-09-18, P0 item 5): a citation
    chip used to show the internal `note_slug` and its raw update date
    inline (e.g. `[signal-pricing-larkspur-regional-health, updated
    2026-08-21]`) -- exactly the kind of engineer-facing detail a design-
    partner exec would ask "what is that" about. `_chat_sentence.html`
    now shows the cited note's own human `title` (from `note_summaries`,
    batched per answer by `routes/chat.py`) as the chip's visible label,
    with the slug and the date moved into the `title=` hover attribute."""
    from portolan.search.embed import FakeEmbedder
    from portolan.search.query import search as run_search
    from portolan.web.queries import note_summaries

    question = "What happened with the account, chip-label edition?"
    result = run_search(demo_conn, FakeEmbedder(), question, k=8, expand=1)
    retrieved_slug = None
    claim_row = None
    for hit in result.hits:
        row = _one(
            demo_conn,
            "SELECT id FROM claims WHERE note_slug = %s ORDER BY id LIMIT 1",
            (hit.note_slug,),
        )
        if row is not None:
            retrieved_slug = hit.note_slug
            claim_row = row
            break
    assert claim_row is not None, "none of the search hits have any claims to cite"
    claim_id = claim_row[0]
    meta = note_summaries(demo_conn, [retrieved_slug])[retrieved_slug]

    canned = {
        "sentences": [
            {
                "text": "This is a grounded sentence citing a real note.",
                "note_slugs": [retrieved_slug],
                "claim_ids": [claim_id],
            }
        ]
    }
    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([canned])
    try:
        r = client.post("/chat/new/stream", data={"question": question}, cookies=auth_cookies)
        assert r.status_code == 200
        # The visible chip label is the note's real title, not the slug,
        # JSON-escaped inside the SSE payload (a signal title's em dash
        # arrives escaped).
        assert json.dumps(f"[{meta['title']}]")[1:-1] in r.text
        events = _parse_sse_events(r.text)
        conversation_id = next(p for name, p in events if name == "done")["conversation_id"]

        # A plain page reload (no JSON/SSE escaping in the way) is the
        # cleaner surface for the exact `title="..."` hover attribute --
        # `_chat_sentence.html` renders identically whether streamed or
        # reloaded (its own module docstring).
        reload_page = client.get(f"/c/{conversation_id}", cookies=auth_cookies)
        assert reload_page.status_code == 200
        assert f"[{meta['title']}]" in reload_page.text
        # The slug (and the update date) are still present -- but only
        # inside the hover `title=` attribute, never as visible text.
        assert f'title="{retrieved_slug} · updated' in reload_page.text
    finally:
        client.app.dependency_overrides.pop(get_llm_dep, None)


def test_chat_uses_chat_model_setting_falling_back_to_strong_model(
    demo_conn: psycopg.Connection, db_url: str, store_root: Path
) -> None:
    """Fix wave 2026-09-13 ("the chat feels dead"): the chat answer path
    used to send no explicit `model` to `llm.complete()`, so it silently
    rode whichever model the `LLMClient` was constructed with --
    `get_llm()` always builds one with `default_model=extraction_model`,
    picked for batch-extraction cost/throughput, not interactive latency
    (on Railway, a ~100s/call model). `settings.chat_model` (env
    `CHAT_MODEL`), when set, must be what chat actually requests; unset,
    it must fall back to `strong_model` -- never to `extraction_model`.
    """
    from portolan.search.query import search as run_search

    question = "What has the account said, model-check edition?"
    assert run_search(demo_conn, FakeEmbedder(), question, k=8, expand=1).hits, (
        "need at least one search hit so answer_question() actually calls the LLM"
    )

    seen_models: list[str | None] = []

    def _capture(*, system, user, json_schema, model, surface):
        del system, user, json_schema, surface
        seen_models.append(model)
        return {"sentences": []}

    def _post_question(chat_model: str | None, strong_model: str) -> None:
        settings = _settings(
            db_url,
            store_root,
            extraction_model="should-never-be-requested-by-chat",
            chat_model=chat_model,
            strong_model=strong_model,
        )
        app = create_app(settings)
        app.dependency_overrides[get_conn] = lambda: demo_conn
        app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
        app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient(_capture)
        with TestClient(app) as c:
            r = c.post(
                "/login",
                data={"org_key": ORG_KEY, "editor": "Model Checker"},
                follow_redirects=False,
            )
            c.post("/chat/new/stream", data={"question": question}, cookies=r.cookies)

    _post_question(chat_model="claude-haiku-4-5", strong_model="claude-sonnet-5")
    assert seen_models == ["claude-haiku-4-5"], "CHAT_MODEL set must be what chat requests"

    seen_models.clear()
    _post_question(chat_model=None, strong_model="claude-sonnet-5")
    assert seen_models == ["claude-sonnet-5"], (
        "CHAT_MODEL unset must fall back to strong_model, not extraction_model"
    )


def test_resolve_chat_model_agent_sdk_defaults_to_claude_sonnet_5_5(
    db_url: str, store_root: Path
) -> None:
    """Owner verdict 2026-09-18 ("make chat use a more capable model"):
    under `llm_provider="agent_sdk"`, `resolve_chat_model` must land on
    Claude no matter what `CHAT_MODEL` happens to hold -- proven against
    the real failure mode this checkout hit: a local `.env` that mirrors
    the Railway deployment (`CHAT_MODEL` unset there, `STRONG_MODEL` an
    OpenRouter "vendor/model" id) with only `LLM_PROVIDER` flipped to
    `agent_sdk` for a local run.
    """
    from portolan.web.routes.chat import DEFAULT_AGENT_SDK_CHAT_MODEL, resolve_chat_model

    assert DEFAULT_AGENT_SDK_CHAT_MODEL == "claude-sonnet-5-5"

    # CHAT_MODEL unset -> the Claude default, not strong_model (which, on
    # this exact checkout's own .env, is an OpenRouter-shaped id).
    unset = _settings(
        db_url,
        store_root,
        llm_provider="agent_sdk",
        chat_model=None,
        strong_model="openai/gpt-5.6-luna",
    )
    assert resolve_chat_model(unset) == "claude-sonnet-5-5"

    # CHAT_MODEL set to a non-Claude (OpenRouter-shaped) id -> still the
    # Claude default, never handed to the Agent SDK as-is.
    non_claude = _settings(
        db_url, store_root, llm_provider="agent_sdk", chat_model="deepseek/deepseek-v4-flash-0731"
    )
    assert resolve_chat_model(non_claude) == "claude-sonnet-5-5"

    # CHAT_MODEL set to a real Claude id -> honored exactly, not
    # overridden by the default.
    claude_set = _settings(
        db_url, store_root, llm_provider="agent_sdk", chat_model="claude-haiku-4-5"
    )
    assert resolve_chat_model(claude_set) == "claude-haiku-4-5"


def test_resolve_chat_model_never_touches_the_deployed_openrouter_path(
    db_url: str, store_root: Path
) -> None:
    """The agent_sdk-only carve-out above must not change behavior for
    any other provider -- especially `openrouter`, the deployed path,
    which routes an OpenRouter "vendor/model" id straight through
    (unchanged from before this fix wave: `chat_model or strong_model`)."""
    from portolan.web.routes.chat import resolve_chat_model

    settings = _settings(
        db_url,
        store_root,
        llm_provider="openrouter",
        chat_model="deepseek/deepseek-v4-flash-0731",
        strong_model="openai/gpt-5.6-luna",
    )
    assert resolve_chat_model(settings) == "deepseek/deepseek-v4-flash-0731"

    unset = _settings(
        db_url,
        store_root,
        llm_provider="openrouter",
        chat_model=None,
        strong_model="openai/gpt-5.6-luna",
    )
    assert resolve_chat_model(unset) == "openai/gpt-5.6-luna"


def test_title_from_question_keeps_the_first_six_words(db_url: str, store_root: Path) -> None:
    """Owner verdict 2026-09-18: the sidebar auto-title reads like a
    short label, not a mid-word 60-character slice."""
    from portolan.web.conversations import title_from_question

    assert (
        title_from_question("What has Larkspur said about renewal?")
        == "What has Larkspur said about renewal?"
    ), "6 words or fewer must not gain a trailing ellipsis"
    assert (
        title_from_question("What has the account said recently about pricing and renewal?")
        == "What has the account said recently…"
    )
    assert title_from_question("   ") == "New chat"
    assert title_from_question("") == "New chat"
    # Extra whitespace between words collapses, same as before this change.
    assert title_from_question("word1   word2") == "word1 word2"


def test_chat_stream_question_over_32000_chars_returns_400_not_500(
    client: TestClient, auth_cookies
) -> None:
    """Review finding F15's bound still applies to the stream endpoint.

    Owner verdict 2026-09-18 ("I don't know why that's a thing"): the cap
    was raised from 4,000 to `Settings.chat_question_max_chars`'s default
    of 32,000 -- still bounded, just no longer a wall a reader pasting in
    a real paragraph could hit by accident.
    """
    r = client.post("/chat/new/stream", data={"question": "x" * 32_001}, cookies=auth_cookies)
    assert r.status_code == 400
    assert "32,000" in r.text


def test_chat_stream_question_at_the_32000_char_limit_is_accepted(
    client: TestClient, auth_cookies
) -> None:
    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([{"sentences": []}])
    r = client.post("/chat/new/stream", data={"question": "x" * 32_000}, cookies=auth_cookies)
    assert r.status_code == 200
    client.app.dependency_overrides.pop(get_llm_dep, None)


def test_chat_stream_question_cap_is_a_settings_field_not_a_hardcoded_constant(
    client: TestClient, demo_conn: psycopg.Connection, db_url: str, store_root: Path, auth_cookies
) -> None:
    """Owner instruction: "make the cap a setting with that default."
    `Settings.chat_question_max_chars` must actually be what the route
    enforces -- proven by overriding it to something small and confirming
    the route's own 400 threshold moves with it, not just that the field
    exists and is ignored."""
    settings = _settings(db_url, store_root, chat_question_max_chars=10)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([{"sentences": []}])
    with TestClient(app) as c:
        login = c.post(
            "/login", data={"org_key": ORG_KEY, "editor": "Cap Tester"}, follow_redirects=False
        )
        ok = c.post("/chat/new/stream", data={"question": "x" * 10}, cookies=login.cookies)
        assert ok.status_code == 200
        too_long = c.post("/chat/new/stream", data={"question": "x" * 11}, cookies=login.cookies)
        assert too_long.status_code == 400
        assert "10" in too_long.text


def test_chat_stream_on_an_existing_conversation_appends_messages(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    created = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    assert created.status_code == 303
    conversation_id = int(created.headers["location"].rsplit("/", 1)[-1])

    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([{"sentences": []}])
    r = client.post(
        f"/chat/{conversation_id}/stream",
        data={"question": "Anything at all?"},
        cookies=auth_cookies,
    )
    assert r.status_code == 200
    client.app.dependency_overrides.pop(get_llm_dep, None)

    with demo_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM messages WHERE conversation_id = %s", (conversation_id,))
        (count,) = cur.fetchone()
    assert count == 2  # one user, one "nothing recorded"/"index broken" assistant message


def test_chat_stream_done_event_for_a_new_conversation_carries_the_sidebar_row(
    client: TestClient, auth_cookies
) -> None:
    """Bug hunt 2026-09-18, finding 7: a brand-new conversation's sidebar
    row used to be a hand-built stub (`chat.html`'s old inline JS) with
    just a title link -- no Rename/Archive/Delete menu, no rename form,
    no delete-confirm block -- until the page was reloaded. The `done`
    event for a `conversation_id == "new"` stream now carries the real
    `_conversation_item.html` render of that row."""
    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([{"sentences": []}])
    r = client.post(
        "/chat/new/stream", data={"question": "does this new row work"}, cookies=auth_cookies
    )
    client.app.dependency_overrides.pop(get_llm_dep, None)
    assert r.status_code == 200

    done_events = [payload for event, payload in _parse_sse_events(r.text) if event == "done"]
    assert len(done_events) == 1
    done = done_events[0]
    assert "conversation_item_html" in done
    html = done["conversation_item_html"]
    assert 'data-action="show-rename"' in html
    assert 'data-action="show-delete"' in html
    assert f'/conversations/{done["conversation_id"]}/archive"' in html


def test_chat_stream_done_event_on_an_existing_conversation_has_no_sidebar_row(
    client: TestClient, auth_cookies
) -> None:
    """The fragment above is only needed the one time there's no
    server-rendered row to fall back on -- every other turn (existing
    conversation) leaves it out, same as before this fix."""
    created = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    conversation_id = int(created.headers["location"].rsplit("/", 1)[-1])

    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([{"sentences": []}])
    r = client.post(
        f"/chat/{conversation_id}/stream",
        data={"question": "a second question"},
        cookies=auth_cookies,
    )
    client.app.dependency_overrides.pop(get_llm_dep, None)

    done_events = [payload for event, payload in _parse_sse_events(r.text) if event == "done"]
    assert len(done_events) == 1
    assert "conversation_item_html" not in done_events[0]


def test_chat_stream_on_someone_elses_conversation_404s(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    with demo_conn.cursor() as cur:
        cur.execute("INSERT INTO conversations (owner) VALUES ('Someone Else') RETURNING id")
        (other_id,) = cur.fetchone()
    demo_conn.commit()

    r = client.post(f"/chat/{other_id}/stream", data={"question": "peeking"}, cookies=auth_cookies)
    assert r.status_code == 404


def test_conversations_module_functions_are_scoped_to_owner(demo_conn: psycopg.Connection) -> None:
    """Review finding 13: `conversation_messages`, `last_user_message`,
    `last_message` and `delete_message` used to take no `owner` at all --
    the module's own docstring claimed every read/write here is scoped,
    which wasn't true of these four. Proven directly (no route in
    between): calling each with the WRONG owner sees/deletes nothing."""
    from portolan.web.conversations import (
        add_message,
        conversation_messages,
        create_conversation,
        delete_message,
        last_message,
        last_user_message,
    )

    conversation = create_conversation(demo_conn, owner="Finding13 Owner")
    add_message(
        demo_conn,
        conversation_id=conversation.id,
        owner="Finding13 Owner",
        role="user",
        content_md="hello",
    )
    message = add_message(
        demo_conn,
        conversation_id=conversation.id,
        owner="Finding13 Owner",
        role="assistant",
        content_md="hi",
    )

    assert conversation_messages(demo_conn, conversation.id, owner="Wrong Owner") == []
    assert last_user_message(demo_conn, conversation.id, owner="Wrong Owner") is None
    assert last_message(demo_conn, conversation.id, owner="Wrong Owner") is None

    # A wrong-owner delete must not remove the row -- proven by then
    # successfully reading it back under the real owner.
    delete_message(demo_conn, message.id, owner="Wrong Owner")
    assert last_message(demo_conn, conversation.id, owner="Finding13 Owner") is not None

    delete_message(demo_conn, message.id, owner="Finding13 Owner")
    remaining = conversation_messages(demo_conn, conversation.id, owner="Finding13 Owner")
    assert [m.role for m in remaining] == ["user"]


def test_add_message_refuses_a_conversation_the_owner_does_not_own(
    demo_conn: psycopg.Connection,
) -> None:
    """`add_message` checks the owner itself, so a caller that skipped
    `get_conversation(..., owner=)` still cannot append to someone else's
    conversation."""
    from portolan.web.conversations import add_message, conversation_messages, create_conversation

    conversation = create_conversation(demo_conn, owner="Append Owner")

    with pytest.raises(LookupError):
        add_message(
            demo_conn,
            conversation_id=conversation.id,
            owner="Not The Owner",
            role="user",
            content_md="appended by someone else",
        )
    with pytest.raises(LookupError):
        add_message(
            demo_conn,
            conversation_id=10**12,
            owner="Append Owner",
            role="user",
            content_md="no such conversation",
        )

    assert conversation_messages(demo_conn, conversation.id, owner="Append Owner") == []
    own = add_message(
        demo_conn,
        conversation_id=conversation.id,
        owner="Append Owner",
        role="user",
        content_md="mine",
    )
    assert own.conversation_id == conversation.id
    stored = conversation_messages(demo_conn, conversation.id, owner="Append Owner")
    assert [m.content_md for m in stored] == ["mine"]


def test_chat_regenerate_replaces_the_last_assistant_message(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    # A question that genuinely retrieves hits, cited by a real note_slug,
    # so `_validate_sentences` (answer.py, untouched by this feature) lets
    # the fake sentence survive instead of falling back to "nothing
    # recorded" -- same guard the other tests in this module use
    # `run_search` for directly.
    from portolan.search.query import search as run_search

    question = "What has the account said about renewal or pricing?"
    result = run_search(demo_conn, FakeEmbedder(), question, k=8, expand=1)
    assert result.hits
    retrieved_slug = result.hits[0].note_slug

    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient(
        [
            {
                "sentences": [
                    {"text": "First answer.", "note_slugs": [retrieved_slug], "claim_ids": []}
                ]
            }
        ]
    )
    r = client.post("/chat/new/stream", data={"question": question}, cookies=auth_cookies)
    assert r.status_code == 200
    conversation_id = _parse_sse_events(r.text)[-1][1]["conversation_id"]
    client.app.dependency_overrides.pop(get_llm_dep, None)

    with demo_conn.cursor() as cur:
        cur.execute(
            "SELECT id, role, content_md FROM messages WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        )
        before_rows = cur.fetchall()
    assert [row[1] for row in before_rows] == ["user", "assistant"]
    old_assistant_id = before_rows[1][0]
    assert "First answer." in before_rows[1][2]

    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient(
        [
            {
                "sentences": [
                    {"text": "Regenerated answer.", "note_slugs": [retrieved_slug], "claim_ids": []}
                ]
            }
        ]
    )
    r2 = client.post(f"/chat/{conversation_id}/regenerate", cookies=auth_cookies)
    assert r2.status_code == 200
    assert "Regenerated answer." in r2.text
    client.app.dependency_overrides.pop(get_llm_dep, None)

    with demo_conn.cursor() as cur:
        cur.execute(
            "SELECT id, role, content_md FROM messages WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        )
        after_rows = cur.fetchall()
    assert [row[1] for row in after_rows] == ["user", "assistant"], (
        "regenerate must replace, not append -- still exactly one user and one assistant message"
    )
    assert after_rows[1][0] != old_assistant_id
    assert "Regenerated answer." in after_rows[1][2]
    assert "First answer." not in after_rows[1][2]


def test_chat_regenerate_requires_session(client: TestClient) -> None:
    r = client.post("/chat/1/regenerate")
    assert r.status_code == 401


# --------------------------------------------------------------------
# Review findings 2, 3, 4: SSE lifecycle -- persist-before-disconnect,
# regenerate ordering, no exception text on the wire.
# --------------------------------------------------------------------


class _FakeRequestForStream:
    """The only thing `_stream_answer` reads off `request`:
    `request.app.state.templates`. `client.app` (below) is the same
    FastAPI app `TestClient` drives, with `.state.templates` already
    built by `create_app` -- no need for a real ASGI `Request`."""

    def __init__(self, app) -> None:
        self.app = app


def test_stream_answer_persists_the_assistant_message_before_the_first_yield(
    client: TestClient, demo_conn: psycopg.Connection
) -> None:
    """Review finding 2, proven the way the reviewer proved it: drive
    `_stream_answer` directly, consume exactly one event, then
    `gen.close()` -- exactly what an ASGI server does to a generator when
    its client disconnects (`GeneratorExit` raised at the current
    `yield`). Before the fix, persistence happened after the last
    `sentence` was yielded, so this left zero assistant messages and zero
    usage rows despite the model having already been called and billed.
    """
    from portolan.search.query import search as run_search
    from portolan.web.conversations import create_conversation
    from portolan.web.routes.chat import _stream_answer

    question = "Disconnect safety: what has the account said?"
    result = run_search(demo_conn, FakeEmbedder(), question, k=8, expand=1)
    assert result.hits
    retrieved_slug = result.hits[0].note_slug

    conversation = create_conversation(demo_conn, owner="Disconnect Tester")
    llm = FakeLLMClient(
        [
            {
                "sentences": [
                    {
                        "text": "Answer before disconnect.",
                        "note_slugs": [retrieved_slug],
                        "claim_ids": [],
                    }
                ]
            }
        ]
    )

    usage_before = _one(demo_conn, "SELECT count(*) FROM usage_events WHERE surface = 'chat'")[0]

    gen = _stream_answer(
        _FakeRequestForStream(client.app),
        demo_conn,
        llm,
        FakeEmbedder(),
        tenant="test",
        editor="Disconnect Tester",
        conversation=conversation,
        question=question,
        model="whatever-model",
    )
    next(gen)  # consume exactly one event -- a client that then vanishes still saw this much
    gen.close()  # what an ASGI server does on disconnect

    with demo_conn.cursor() as cur:
        cur.execute(
            "SELECT role FROM messages WHERE conversation_id = %s ORDER BY id", (conversation.id,)
        )
        roles = [row[0] for row in cur.fetchall()]
    assert roles == ["assistant"], (
        "the assistant message must already be written before a disconnect can lose it"
    )

    usage_after = _one(demo_conn, "SELECT count(*) FROM usage_events WHERE surface = 'chat'")[0]
    assert usage_after == usage_before + 1


def test_chat_regenerate_preserves_the_old_answer_when_the_model_fails(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding 3: regenerate used to delete the trailing assistant
    message before attempting the replacement, so a transient model
    failure destroyed the reader's previous answer for nothing. Proven
    directly: monkeypatch `answer_question` (as imported into
    `routes.chat`) to raise, then regenerate and confirm the earlier
    answer is untouched in the database."""
    from portolan.search.query import search as run_search

    question = "What has the account said about renewal or pricing, take two?"
    result = run_search(demo_conn, FakeEmbedder(), question, k=8, expand=1)
    assert result.hits
    retrieved_slug = result.hits[0].note_slug

    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient(
        [{"sentences": [{"text": "Kept answer.", "note_slugs": [retrieved_slug], "claim_ids": []}]}]
    )
    r = client.post("/chat/new/stream", data={"question": question}, cookies=auth_cookies)
    assert r.status_code == 200
    conversation_id = _parse_sse_events(r.text)[-1][1]["conversation_id"]
    client.app.dependency_overrides.pop(get_llm_dep, None)

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic model failure for the regenerate-preserves test")

    monkeypatch.setattr("portolan.web.routes.chat.answer_question", _boom)

    r2 = client.post(f"/chat/{conversation_id}/regenerate", cookies=auth_cookies)
    assert r2.status_code == 200
    events = _parse_sse_events(r2.text)
    assert events and events[0][0] == "error"

    with demo_conn.cursor() as cur:
        cur.execute(
            "SELECT role, content_md FROM messages WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        )
        rows = cur.fetchall()
    assert [row[0] for row in rows] == ["user", "assistant"], (
        "a failed regenerate must not have deleted the previous assistant message"
    )
    assert "Kept answer." in rows[1][1]


def test_chat_stream_error_never_leaks_the_exception_text(
    client: TestClient, auth_cookies, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Review finding 4: the SSE error path used to stream `str(exc)`
    straight to the browser -- `answer_question` can fail inside
    `search()` (a psycopg error carrying the failing SQL plus the
    connection's host/user/database) or `llm.complete()` (an upstream
    provider's raw error body). Plants a distinctive marker in the raised
    exception's message and asserts it never reaches the response, only
    the one fixed sentence -- and that the real exception is still logged
    server-side."""
    import logging

    from portolan.web.routes import chat as chat_module

    marker = "synthetic-secret-marker-should-never-reach-the-browser"

    def _boom(*args, **kwargs):
        raise RuntimeError(marker)

    monkeypatch.setattr(chat_module, "answer_question", _boom)

    with caplog.at_level(logging.ERROR, logger=chat_module.__name__):
        r = client.post(
            "/chat/new/stream", data={"question": "trigger the error path"}, cookies=auth_cookies
        )
    assert r.status_code == 200
    assert marker not in r.text
    assert chat_module.GENERIC_ANSWER_ERROR in r.text
    assert any(marker in record.getMessage() for record in caplog.records) or any(
        marker in str(record.exc_info) for record in caplog.records if record.exc_info
    ), "the real exception must still be logged server-side"


# --------------------------------------------------------------------
# Review finding 8: an empty/whitespace question is rejected before a
# conversation is created or the model is called.
# --------------------------------------------------------------------


def test_chat_stream_empty_question_is_rejected_before_a_conversation_is_created(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    before = _one(demo_conn, "SELECT count(*) FROM conversations")[0]
    usage_before = _one(demo_conn, "SELECT count(*) FROM usage_events WHERE surface = 'chat'")[0]

    r = client.post("/chat/new/stream", data={"question": "   "}, cookies=auth_cookies)
    assert r.status_code == 400

    after = _one(demo_conn, "SELECT count(*) FROM conversations")[0]
    usage_after = _one(demo_conn, "SELECT count(*) FROM usage_events WHERE surface = 'chat'")[0]
    assert after == before, "an empty question must not create a conversation"
    assert usage_after == usage_before, "an empty question must never reach the model"


# --------------------------------------------------------------------
# Review finding 7: a non-numeric claim_id on the edit form is a 400.
# --------------------------------------------------------------------


def test_edit_form_non_numeric_claim_id_returns_400_not_500(
    client: TestClient, auth_cookies, any_claim
) -> None:
    note_slug, *_ = any_claim
    r = client.post(
        f"/notes/{note_slug}/edit",
        data={"after_text": "x", "claim_id": "abc"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r.status_code == 400


# --------------------------------------------------------------------
# Conversations: CRUD, ownership, display-name-required login
# --------------------------------------------------------------------


def test_login_requires_a_display_name(client: TestClient) -> None:
    r = client.post("/login", data={"org_key": ORG_KEY, "editor": ""})
    assert r.status_code == 400


def test_login_display_name_too_short_rejected(client: TestClient) -> None:
    r = client.post("/login", data={"org_key": ORG_KEY, "editor": "A"})
    assert r.status_code == 400


def test_login_display_name_too_long_rejected(client: TestClient) -> None:
    r = client.post("/login", data={"org_key": ORG_KEY, "editor": "x" * 41})
    assert r.status_code == 400


def test_login_display_name_at_bounds_is_accepted(client: TestClient) -> None:
    r = client.post("/login", data={"org_key": ORG_KEY, "editor": "Al"}, follow_redirects=False)
    assert r.status_code == 303
    r2 = client.post(
        "/login", data={"org_key": ORG_KEY, "editor": "x" * 40}, follow_redirects=False
    )
    assert r2.status_code == 303


def test_session_carries_the_display_name_as_conversation_owner(
    client: TestClient, demo_conn: psycopg.Connection
) -> None:
    r = client.post(
        "/login", data={"org_key": ORG_KEY, "editor": "Priya Owner"}, follow_redirects=False
    )
    assert r.status_code == 303
    cookies = r.cookies
    created = client.post("/conversations", cookies=cookies, follow_redirects=False)
    assert created.status_code == 303
    conversation_id = int(created.headers["location"].rsplit("/", 1)[-1])
    owner = _one(demo_conn, "SELECT owner FROM conversations WHERE id = %s", (conversation_id,))[0]
    assert owner == "Priya Owner"


def test_conversations_create_redirects_to_the_new_conversation(
    client: TestClient, auth_cookies
) -> None:
    r = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/c/")


def test_conversation_page_requires_session(client: TestClient, auth_cookies) -> None:
    created = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    conversation_id = int(created.headers["location"].rsplit("/", 1)[-1])
    # `client` is a stateful cookie jar and `auth_cookies` already logged in
    # on it -- drop the cookie explicitly rather than merely omitting
    # `cookies=` (same gotcha `test_page_200_with_session_302_without`
    # documents).
    client.cookies.clear()
    r = client.get(f"/c/{conversation_id}", follow_redirects=False)
    assert r.status_code == 302


def test_conversation_page_404s_for_a_nonexistent_conversation(
    client: TestClient, auth_cookies
) -> None:
    r = client.get("/c/99999999", cookies=auth_cookies)
    assert r.status_code == 404


def test_conversation_page_404s_for_another_owners_conversation(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO conversations (owner, title) VALUES ('Not You', 'Their chat') RETURNING id"
        )
        (other_id,) = cur.fetchone()
    demo_conn.commit()
    r = client.get(f"/c/{other_id}", cookies=auth_cookies)
    assert r.status_code == 404


def test_conversations_rename(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    created = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    conversation_id = int(created.headers["location"].rsplit("/", 1)[-1])

    r = client.post(
        f"/conversations/{conversation_id}/rename",
        data={"title": "A better title"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r.status_code == 303
    title = _one(demo_conn, "SELECT title FROM conversations WHERE id = %s", (conversation_id,))[0]
    assert title == "A better title"

    # An HTMX request gets a re-rendered <li> fragment back instead of a
    # redirect, so the sidebar's inline rename can swap it in place.
    r2 = client.post(
        f"/conversations/{conversation_id}/rename",
        data={"title": "Renamed again"},
        cookies=auth_cookies,
        headers={"HX-Request": "true"},
    )
    assert r2.status_code == 200
    assert "Renamed again" in r2.text
    assert f'data-id="{conversation_id}"' in r2.text


def test_conversations_rename_keeps_the_active_highlight_when_renaming_the_open_conversation(
    client: TestClient, auth_cookies
) -> None:
    """Review finding 17: renaming (or archiving) the conversation you're
    currently reading used to swap in a row rendered with a hardcoded
    `active_id=None`, dropping `class="conv-item active"` from it even
    though it's still the open conversation. `_conversation_item.html`'s
    forms now round-trip `current_conversation_id` (the page's own
    `active_id`) back to the server on the HTMX POST."""
    created = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    conversation_id = int(created.headers["location"].rsplit("/", 1)[-1])

    r = client.post(
        f"/conversations/{conversation_id}/rename",
        data={"title": "Renamed while open", "current_conversation_id": str(conversation_id)},
        cookies=auth_cookies,
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    assert "conv-item active" in r.text

    # Renaming a DIFFERENT row than the one currently open must not mark
    # it active -- current_conversation_id names the open conversation,
    # not the one being mutated.
    other = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    other_id = int(other.headers["location"].rsplit("/", 1)[-1])
    r2 = client.post(
        f"/conversations/{other_id}/rename",
        data={"title": "Renamed elsewhere", "current_conversation_id": str(conversation_id)},
        cookies=auth_cookies,
        headers={"HX-Request": "true"},
    )
    assert r2.status_code == 200
    assert "conv-item active" not in r2.text


def test_conversations_rename_another_owners_conversation_404s(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    with demo_conn.cursor() as cur:
        cur.execute("INSERT INTO conversations (owner) VALUES ('Not You') RETURNING id")
        (other_id,) = cur.fetchone()
    demo_conn.commit()
    r = client.post(
        f"/conversations/{other_id}/rename", data={"title": "Hijacked"}, cookies=auth_cookies
    )
    assert r.status_code == 404


def test_conversations_archive_and_unarchive(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    created = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    conversation_id = int(created.headers["location"].rsplit("/", 1)[-1])

    r = client.post(
        f"/conversations/{conversation_id}/archive", cookies=auth_cookies, follow_redirects=False
    )
    assert r.status_code == 303
    archived = _one(
        demo_conn, "SELECT archived FROM conversations WHERE id = %s", (conversation_id,)
    )[0]
    assert archived is True

    # An archived conversation still appears (nothing here is hard-deleted
    # by archiving), but under the "Archived" disclosure, with an
    # "unarchive" action -- not in the active list with an "archive" one.
    home = client.get("/", cookies=auth_cookies)
    assert f'/conversations/{conversation_id}/unarchive"' in home.text
    assert f'/conversations/{conversation_id}/archive"' not in home.text

    r2 = client.post(
        f"/conversations/{conversation_id}/unarchive", cookies=auth_cookies, follow_redirects=False
    )
    assert r2.status_code == 303
    archived_after = _one(
        demo_conn, "SELECT archived FROM conversations WHERE id = %s", (conversation_id,)
    )[0]
    assert archived_after is False


def test_archive_and_unarchive_via_htmx_return_hx_refresh(client: TestClient, auth_cookies) -> None:
    """Bug hunt 2026-09-18, finding 9: an HTMX archive/unarchive used to
    swap only the one `<li>` -- the *other* list, and the "Archived (N)"
    count, stayed stale until a manual reload (archiving never added the
    row under Archived; unarchiving put it back in place *inside* the
    archived list instead of moving it out). `HX-Refresh` tells htmx to
    do a full, correct reload instead."""
    created = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    conversation_id = int(created.headers["location"].rsplit("/", 1)[-1])

    r = client.post(
        f"/conversations/{conversation_id}/archive",
        cookies=auth_cookies,
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    assert r.headers.get("HX-Refresh") == "true"

    r2 = client.post(
        f"/conversations/{conversation_id}/unarchive",
        cookies=auth_cookies,
        headers={"HX-Request": "true"},
    )
    assert r2.status_code == 200
    assert r2.headers.get("HX-Refresh") == "true"


def test_conversations_delete_removes_conversation_and_cascades_messages(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    client.app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([{"sentences": []}])
    r = client.post("/chat/new/stream", data={"question": "to be deleted"}, cookies=auth_cookies)
    conversation_id = _parse_sse_events(r.text)[-1][1]["conversation_id"]
    client.app.dependency_overrides.pop(get_llm_dep, None)

    r = client.post(
        f"/conversations/{conversation_id}/delete", cookies=auth_cookies, follow_redirects=False
    )
    assert r.status_code == 303
    assert _one(demo_conn, "SELECT id FROM conversations WHERE id = %s", (conversation_id,)) is None
    with demo_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM messages WHERE conversation_id = %s", (conversation_id,))
        (count,) = cur.fetchone()
    assert count == 0


def test_conversations_delete_another_owners_conversation_404s(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    with demo_conn.cursor() as cur:
        cur.execute("INSERT INTO conversations (owner) VALUES ('Not You') RETURNING id")
        (other_id,) = cur.fetchone()
    demo_conn.commit()
    r = client.post(f"/conversations/{other_id}/delete", cookies=auth_cookies)
    assert r.status_code == 404
    still_there = _one(demo_conn, "SELECT id FROM conversations WHERE id = %s", (other_id,))
    assert still_there is not None


def test_chat_home_shows_example_questions_from_top_accounts(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    top_account = _one(
        demo_conn, "SELECT title FROM notes WHERE type = 'account' ORDER BY updated DESC LIMIT 1"
    )[0]
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    assert top_account in r.text


def test_chat_home_sidebar_lists_conversations_newest_first(
    client: TestClient, auth_cookies
) -> None:
    first = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    first_id = int(first.headers["location"].rsplit("/", 1)[-1])
    client.post(
        f"/conversations/{first_id}/rename", data={"title": "Older chat"}, cookies=auth_cookies
    )
    second = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    second_id = int(second.headers["location"].rsplit("/", 1)[-1])
    client.post(
        f"/conversations/{second_id}/rename", data={"title": "Newer chat"}, cookies=auth_cookies
    )

    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    assert r.text.index("Newer chat") < r.text.index("Older chat")


def test_chat_sidebar_discloses_history_is_shared_by_display_name(
    client: TestClient, auth_cookies
) -> None:
    """Review finding 10: a conversation's owner is a self-asserted
    display name, not a per-person account -- two readers who type the
    same name at login share one chat history. Per instruction, v1 does
    not build accounts or session ids; instead the sidebar says so in one
    line, on both the empty chat home and a conversation page."""
    home = client.get("/", cookies=auth_cookies)
    assert home.status_code == 200
    assert "shared by everyone who logs in with this same name" in home.text

    created = client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    conversation_id = int(created.headers["location"].rsplit("/", 1)[-1])
    conv_page = client.get(f"/c/{conversation_id}", cookies=auth_cookies)
    assert conv_page.status_code == 200
    assert "shared by everyone who logs in with this same name" in conv_page.text


# --------------------------------------------------------------------
# Notes / sources
# --------------------------------------------------------------------


def test_notes_page_requires_session(client: TestClient, any_claim) -> None:
    note_slug, *_ = any_claim
    r = client.get(f"/notes/{note_slug}", follow_redirects=False)
    assert r.status_code == 302


def test_note_page_renders_claims_with_citation_links(
    client: TestClient, auth_cookies, any_claim
) -> None:
    note_slug, claim_id, source_slug, locator = any_claim
    r = client.get(f"/notes/{note_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert f'id="claim-{claim_id}"' in r.text
    assert f"/sources/{source_slug}" in r.text
    assert f"/originals/{source_slug}?at={quote(locator, safe='/')}" in r.text
    assert "Edit" in r.text


def test_note_page_404_for_missing_note(client: TestClient, auth_cookies) -> None:
    r = client.get("/notes/no-such-note-at-all", cookies=auth_cookies)
    assert r.status_code == 404


# --------------------------------------------------------------------
# Owner report, 2026-09-15: `/notes/{slug}` (any note type, not just a
# report) rendered its synthesis as escaped plain text -- `### Headline`,
# `- [[slug]]: ...` bullets all showed up literally, while `/reports/
# {slug}` already rendered the equivalent field correctly. `render_note_
# body` (`portolan.web.markdown`) is now the one renderer `note_view.
# render_note_detail`, `routes.reports.report_detail` and `routes.graph.
# graph_note_json` all share.
# --------------------------------------------------------------------


def test_note_page_renders_markdown_wikilinks_and_escapes_script(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """A brand-new, throwaway note (`write_note` directly, the same
    pattern `_write_test_report` below uses) rather than editing an
    existing demo note through `/notes/{slug}/edit`: `any_claim` and
    `note_with_links_both_directions` both key off "the first claim's
    note"/"a note with links in both directions" in this session-scoped
    `demo_conn`, so a destructive edit against one of those notes here
    would leave a different, unrelated test reading back a note whose
    synthesis/related section this test just rewrote."""
    from datetime import UTC, datetime

    from portolan.notes.models import NoteRow, NoteStatus, NoteType
    from portolan.notes.persist import write_note

    row = _one(demo_conn, "SELECT slug FROM notes LIMIT 1")
    assert row is not None
    (known_slug,) = row

    now = datetime.now(UTC)
    test_slug = "note-markdown-render-test"
    body = f"### Heading\n\n- [[{known_slug}]]: a wikilink.\n\n<script>alert(1)</script>"
    note = NoteRow(
        slug=test_slug,
        type=NoteType.SIGNAL,
        title="Markdown render test note",
        description="Fixture note for markdown-rendering test coverage.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis=body,
        related=[],
    )
    write_note(
        demo_conn,
        note,
        [],
        changed_by="test",
        reason="markdown render test",
        embedder=FakeEmbedder(),
    )

    detail = client.get(f"/notes/{test_slug}", cookies=auth_cookies)
    assert detail.status_code == 200
    assert "<h3>Heading</h3>" in detail.text
    assert f'<a href="/notes/{known_slug}">' in detail.text
    assert "[[" not in detail.text
    assert "<script>" not in detail.text
    assert "&lt;script&gt;" in detail.text


def test_source_page_links_to_original(client: TestClient, auth_cookies, any_claim) -> None:
    _note_slug, _claim_id, source_slug, _locator = any_claim
    r = client.get(f"/sources/{source_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert f"/originals/{source_slug}" in r.text


# --------------------------------------------------------------------
# Originals viewer
# --------------------------------------------------------------------


def test_original_call_viewer_has_at_anchor(client: TestClient, auth_cookies, call_source) -> None:
    slug, locator = call_source
    r = client.get(f"/originals/{slug}", params={"at": locator}, cookies=auth_cookies)
    assert r.status_code == 200
    assert f'data-locator="{locator}"' in r.text
    assert "getElementById" in r.text or "querySelectorAll" in r.text  # the seek-to-locator island


def test_original_email_viewer_renders_paragraphs(
    client: TestClient, auth_cookies, email_source
) -> None:
    slug, locator = email_source
    r = client.get(f"/originals/{slug}", params={"at": locator}, cookies=auth_cookies)
    assert r.status_code == 200
    assert f'data-locator="{locator}"' in r.text
    assert "loc-block" in r.text


def test_original_ticket_viewer_renders_comments(
    client: TestClient, auth_cookies, ticket_source
) -> None:
    slug, locator = ticket_source
    r = client.get(f"/originals/{slug}", params={"at": locator}, cookies=auth_cookies)
    assert r.status_code == 200
    assert f'data-locator="{locator}"' in r.text


def test_versioned_slug_locators_match_the_base_source_id_extraction_used() -> None:
    # Bug hunt 2026-09-18, finding 10: a re-crawled source's slug is
    # `<source_id>--v<n>` (provenance-chain.md "Source versioning"), but
    # extraction always locator-ed against the connector's own base
    # `source_id`, never the versioned slug -- claims cite
    # "S-ticket-900#1", not "S-ticket-900--v2#1". The viewer used to
    # rebuild its `RawSource` from `slug` directly, so every `?at=`
    # anchor it rendered for a versioned source missed the claim's own
    # locator. A direct unit test against `_build_view`/`_load_source_
    # row`'s shape (rather than the full `demo_conn` fixture, which plants
    # no versioned source) is the cheapest way to pin this down.
    from portolan.web.routes.originals import _build_view

    source_row = {
        "slug": "S-ticket-900--v2",
        "source_id": "S-ticket-900",
        "source_type": "ticket",
        "s3_key": "unused",
        "source_date": date(2026, 8, 1),
        "participants": [],
        "summary": None,
    }
    raw_bytes = json.dumps({"comments": [{"number": 1, "body": "hello"}]}).encode("utf-8")

    view = _build_view(source_row, raw_bytes)

    assert view["blocks"][0]["locator"] == "S-ticket-900#1"
    assert view["blocks"][0]["locator"] != "S-ticket-900--v2#1"


def test_opening_original_logs_usage_event(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, call_source
) -> None:
    slug, _locator = call_source
    before = _one(
        demo_conn, "SELECT count(*) FROM usage_events WHERE extra ->> 'opened' = %s", (slug,)
    )[0]

    r = client.get(f"/originals/{slug}", cookies=auth_cookies)
    assert r.status_code == 200

    after = _one(
        demo_conn, "SELECT count(*) FROM usage_events WHERE extra ->> 'opened' = %s", (slug,)
    )[0]
    assert after == before + 1


def test_original_requires_session(client: TestClient, call_source) -> None:
    slug, _ = call_source
    r = client.get(f"/originals/{slug}", follow_redirects=False)
    assert r.status_code == 302


def test_original_raw_bytes_sets_nosniff(client: TestClient, auth_cookies, call_source) -> None:
    slug, _ = call_source
    r = client.get(f"/originals/{slug}/raw", cookies=auth_cookies)
    assert r.status_code == 200
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_original_raw_bytes_404s_when_the_stored_file_is_missing(
    client: TestClient, auth_cookies, call_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding 18: `/originals/{slug}/raw` used to 500
    (`store.get`'s bare `OSError` escaping uncaught) on exactly the
    condition its sibling `/originals/{slug}` already gives a clean 404
    for -- same fix, same `try/except OSError`."""
    from portolan.store import LocalStore

    slug, _ = call_source

    def _boom(self, key):
        raise OSError("synthetic missing-file failure for the raw-bytes 404 test")

    monkeypatch.setattr(LocalStore, "get", _boom)
    r = client.get(f"/originals/{slug}/raw", cookies=auth_cookies)
    assert r.status_code == 404


def test_original_detail_with_a_nul_byte_in_the_slug_is_a_404_not_a_500(
    client: TestClient, auth_cookies
) -> None:
    """Bug hunt 2026-09-18, finding 6: a NUL byte in `slug` used to reach
    `_load_source_row`'s `slug = %s` query unfiltered -- Postgres text
    fields can't hold a NUL byte at all, so psycopg raised `DataError`
    and the request 500'd. A NUL byte can never be a real slug, so
    `_load_source_row` now rejects it up front as "no such source" --
    the same 404 every other bogus slug already gets."""
    r = client.get("/originals/S-does-not-exist%00", cookies=auth_cookies)
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")


def test_original_raw_bytes_with_a_nul_byte_in_the_slug_is_a_404_not_a_500(
    client: TestClient, auth_cookies
) -> None:
    r = client.get("/originals/S-does-not-exist%00/raw", cookies=auth_cookies)
    assert r.status_code == 404


@pytest.fixture()
def pdf_source(demo_conn: psycopg.Connection, store_root: Path) -> str:
    """Bug hunt 2026-09-18, finding 4's own repro note: the synthetic
    demo corpus has no `doc` sources at all, so the one PDF-viewer code
    path (`original_pdf.html`'s iframe, `_pdf_page_from_locator`) is
    otherwise unreachable by any test in this file. Inserts one directly
    -- a minimal `sources` row plus real `%PDF-` bytes in the store --
    exactly the hunter's own repro shape."""
    slug = "S-doc-bughunt-test"
    s3_key = f"{slug}.pdf"
    with demo_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, source_date)
            VALUES (%s, 'doc', %s, '2026-03-02')
            ON CONFLICT (slug) DO UPDATE SET s3_key = EXCLUDED.s3_key
            """,
            (slug, s3_key),
        )
    demo_conn.commit()
    LocalStore(store_root).put(s3_key, b"%PDF-1.4\n%mock pdf for tests\n", "application/pdf")
    return slug


def test_pdf_original_viewer_frame_ancestors_allows_same_origin_embedding(
    client: TestClient, auth_cookies, pdf_source: str
) -> None:
    """Bug hunt 2026-09-18, finding 4: every response -- including
    `/originals/{slug}/raw`, the exact response `original_pdf.html`'s own
    `<iframe>` embeds -- used to carry `Content-Security-Policy:
    ... frame-ancestors 'none'`, which forbids embedding by *any* origin,
    same-origin included. The one source type (`doc`, when it's a PDF)
    the iframe/seek-to-locator code was written for was broken by
    construction. `frame-ancestors 'self'` still blocks every other site
    from framing this app (clickjacking) while letting the app frame its
    own pages."""
    page = client.get(f"/originals/{pdf_source}", cookies=auth_cookies)
    assert page.status_code == 200
    assert f'src="/originals/{pdf_source}/raw' in page.text
    assert "frame-ancestors 'none'" not in page.headers.get("content-security-policy", "")

    raw = client.get(f"/originals/{pdf_source}/raw", cookies=auth_cookies)
    assert raw.status_code == 200
    assert raw.headers["content-type"] == "application/pdf"
    csp = raw.headers.get("content-security-policy", "")
    assert "frame-ancestors 'none'" not in csp
    assert "frame-ancestors 'self'" in csp


def test_rss_link_with_a_javascript_scheme_renders_as_text_not_a_link(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, store_root: Path
) -> None:
    """Review finding F5, proven end to end against a live app: a stored
    RSS original's `link` set to `javascript:alert(document.cookie)` was
    emitted verbatim into an `href` -- autoescaping stops attribute
    break-out, not the scheme. `templating._safe_href` now scheme-checks
    any href built from vendor/model data; anything outside http/https/
    mailto renders as inert text instead."""
    import json

    row = _one(demo_conn, "SELECT slug, s3_key FROM sources WHERE source_type = 'rss' LIMIT 1")
    assert row is not None, "demo graph has no RSS source"
    slug, s3_key = row

    hostile_payload = json.dumps(
        {
            "id": slug,
            "date": "2026-03-02",
            "title": "Hostile link test",
            "link": "javascript:alert(document.cookie)",
            "summary": "irrelevant",
        }
    ).encode("utf-8")
    LocalStore(store_root).put(s3_key, hostile_payload, "application/json")

    r = client.get(f"/originals/{slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert 'href="javascript:' not in r.text
    # still shown, just not as a clickable link
    assert "javascript:alert(document.cookie)" in r.text


# --------------------------------------------------------------------
# Edit
# --------------------------------------------------------------------


def test_edit_requires_session(client: TestClient, any_claim) -> None:
    note_slug, claim_id, *_ = any_claim
    r = client.post(
        f"/notes/{note_slug}/edit",
        data={"after_text": "x", "claim_id": str(claim_id)},
    )
    assert r.status_code == 401


def test_edit_route_passes_the_new_text_to_record_edit_without_pre_applying(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Review finding F16 / the wave-2 coordination point. `routes/notes.py`
    no longer pre-applies `after_text` itself (no direct `UPDATE claims`,
    no `write_note` call before `record_edit` runs) -- it computes
    `before_text` from the note's current state and calls `record_edit`
    once, passing `after=after_text` straight through, and lets
    `record_edit` (now the single writer -- `portolan/consolidate/
    edits.py`'s own fix for the matching finding, G11) apply it, append
    the citation marker, and write the note exactly once: one new
    `edit_records` row, one new `note_versions` row (not two -- the
    review's proven F16 defect was 4 -> 6 for a single synthesis edit),
    and the rendered note showing the real typed text plus the marker.
    """
    row = _one(
        demo_conn,
        "SELECT note_slug, id, claim_text FROM claims "
        "WHERE claim_text NOT LIKE %s ORDER BY id LIMIT 1",
        ("%^[edit:%",),
    )
    assert row is not None
    note_slug, claim_id, before_text = row

    before_edits = _one(
        demo_conn, "SELECT count(*) FROM edit_records WHERE note_slug = %s", (note_slug,)
    )[0]
    before_versions = _note_version(demo_conn, note_slug)

    # No apostrophe: Jinja autoescapes it to &#39; in the rendered page,
    # which would break a plain substring check against detail.text below.
    after_text = "Edited via the web app edit form."
    r = client.post(
        f"/notes/{note_slug}/edit",
        data={"after_text": after_text, "reason": "test coverage", "claim_id": str(claim_id)},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/notes/{note_slug}"

    after_edits, edit_before, edit_after = _one(
        demo_conn,
        "SELECT count(*), max(before_text), max(after_text) FROM edit_records WHERE note_slug = %s",
        (note_slug,),
    )
    assert after_edits == before_edits + 1
    assert edit_before == before_text
    # record_edit applies after_text plus the citation marker as the
    # single writer -- edit_records.after_text is what actually got
    # written, not the bare form value.
    assert edit_after.startswith(after_text)
    assert "^[edit:" in edit_after

    after_versions = _note_version(demo_conn, note_slug)
    assert after_versions == before_versions + 1, (
        "one edit should cost exactly one note_versions row (review finding F16), "
        f"got {before_versions} -> {after_versions}"
    )

    detail = client.get(f"/notes/{note_slug}", cookies=auth_cookies)
    assert detail.status_code == 200
    assert "^[edit:" in detail.text
    assert after_text in detail.text


def test_edit_synthesis_costs_exactly_one_note_version(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    from portolan.notes.persist import read_note

    row = _one(demo_conn, "SELECT slug FROM notes WHERE type = 'account' LIMIT 1")
    assert row is not None
    (slug,) = row
    before_note, _before_claims = read_note(demo_conn, slug)
    before_synthesis = before_note.synthesis or ""

    before_versions = _note_version(demo_conn, slug)

    after_text = "A hand-edited synthesis for testing."
    r = client.post(
        f"/notes/{slug}/edit",
        data={"after_text": after_text, "reason": "test"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r.status_code == 303

    edit_before, edit_after = _one(
        demo_conn,
        "SELECT before_text, after_text FROM edit_records WHERE note_slug = %s "
        "ORDER BY id DESC LIMIT 1",
        (slug,),
    )
    assert edit_before == before_synthesis
    assert edit_after.startswith(after_text)
    assert "^[edit:" in edit_after

    after_versions = _note_version(demo_conn, slug)
    assert after_versions == before_versions + 1

    detail = client.get(f"/notes/{slug}", cookies=auth_cookies)
    assert "^[edit:" in detail.text
    assert after_text in detail.text


# --------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------


def test_graph_page_requires_session(client: TestClient) -> None:
    r = client.get("/graph", follow_redirects=False)
    assert r.status_code == 302


def test_graph_json_requires_session(client: TestClient) -> None:
    r = client.get("/graph.json")
    assert r.status_code == 401


def test_graph_json_returns_typed_nodes(client: TestClient, auth_cookies) -> None:
    r = client.get("/graph.json", cookies=auth_cookies)
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) > 0
    assert all({"id", "title", "type", "updated"} <= node.keys() for node in data["nodes"])
    assert len(data["nodes"]) <= 2000


def test_graph_json_type_filter(client: TestClient, auth_cookies) -> None:
    r = client.get("/graph.json", params={"type": "account"}, cookies=auth_cookies)
    assert r.status_code == 200
    data = r.json()
    assert data["nodes"]
    assert all(node["type"] == "account" for node in data["nodes"])


# --------------------------------------------------------------------
# /graph/note/{slug}.json -- the graph viewer's note panel (restyle wave
# 2026-09-15 v2, build instructions item 5).
# --------------------------------------------------------------------


def test_graph_note_json_requires_session(
    client: TestClient, demo_conn: psycopg.Connection
) -> None:
    row = _one(demo_conn, "SELECT slug FROM notes LIMIT 1")
    assert row is not None
    r = client.get(f"/graph/note/{row[0]}.json")
    assert r.status_code == 401


def test_graph_note_json_404_on_unknown_slug(client: TestClient, auth_cookies) -> None:
    r = client.get("/graph/note/does-not-exist-anywhere.json", cookies=auth_cookies)
    assert r.status_code == 404


def test_graph_note_json_returns_the_expected_shape(
    client: TestClient, auth_cookies, demo_conn: psycopg.Connection
) -> None:
    row = _one(demo_conn, "SELECT slug, title, type FROM notes LIMIT 1")
    assert row is not None
    slug, title, note_type = row
    r = client.get(f"/graph/note/{slug}.json", cookies=auth_cookies)
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"slug", "title", "type", "updated", "summary", "body_html", "links"}
    assert data["slug"] == slug
    assert data["title"] == title
    assert data["type"] == note_type
    assert data["summary"]  # never empty -- _summary_text falls back to the title
    assert isinstance(data["links"], list)
    assert all({"slug", "title", "type"} <= link.keys() for link in data["links"])


def test_graph_note_json_links_both_directions(
    client: TestClient, auth_cookies, note_with_links_both_directions: tuple[str, str, str]
) -> None:
    """A note's `links` is the union of its outgoing (`note.related`, body/
    parent/supersession links parsed out of `body_md`) and incoming
    (`backlinks()`) edges -- both directions in one deduplicated list."""
    slug, outgoing_slug, incoming_slug = note_with_links_both_directions
    r = client.get(f"/graph/note/{slug}.json", cookies=auth_cookies)
    assert r.status_code == 200
    linked_slugs = {item["slug"] for item in r.json()["links"]}
    assert outgoing_slug in linked_slugs
    assert incoming_slug in linked_slugs


# --------------------------------------------------------------------
# Graph viewer: 2D/3D mode, pinned CDN versions, degree, cap notice
# (UI wave 1, item 3)
# --------------------------------------------------------------------


def test_graph_page_defaults_to_3d_mode(client: TestClient, auth_cookies) -> None:
    """Restyle wave 2026-09-15 (build instructions "Motion"): 3D is now
    the default view, auto-rotating from the start -- a bare `/graph`
    (no `mode` query param) must render the 3D island's initial JS state,
    not fall back to the old 2D default."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    assert 'mode: "3d",' in r.text
    assert 'id="mode-3d" class="btn" aria-pressed="true"' in r.text


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_graph_page_accepts_a_mode_parameter(client: TestClient, auth_cookies, mode: str) -> None:
    r = client.get("/graph", params={"mode": mode}, cookies=auth_cookies)
    assert r.status_code == 200


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_graph_json_accepts_a_mode_parameter(client: TestClient, auth_cookies, mode: str) -> None:
    r = client.get("/graph.json", params={"mode": mode}, cookies=auth_cookies)
    assert r.status_code == 200


def test_graph_json_rejects_an_unknown_mode(client: TestClient, auth_cookies) -> None:
    r = client.get("/graph.json", params={"mode": "bogus"}, cookies=auth_cookies)
    assert r.status_code == 400


def test_graph_page_ships_pinned_exact_cdn_versions_for_both_islands(
    client: TestClient, auth_cookies
) -> None:
    """Fix wave 2026-09-12 (var/graph-fix/repro.md): the hand-written d3
    viewer is replaced with two same-author, same-API-shape libraries --
    force-graph (2D, canvas, bundles d3-force internally -- no separate
    d3 script) and 3d-force-graph (3D, unchanged version). Both load from
    pinned jsdelivr URLs with an exact version, never `@latest` or an
    unversioned path; force-graph has no cdnjs listing at all (checked
    against the cdnjs API), so jsdelivr is the only correct host for it."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    assert "cdn.jsdelivr.net/npm/force-graph@1.51.4/dist/force-graph.min.js" in r.text
    assert "cdn.jsdelivr.net/npm/3d-force-graph@1.80.0/dist/3d-force-graph.min.js" in r.text
    assert "@latest" not in r.text
    # the old d3 script is gone -- force-graph bundles d3-force itself
    assert "d3.min.js" not in r.text
    # Review finding 9: a pinned version is not an integrity check -- both
    # islands' <script> tags must carry a real sha384 integrity hash and
    # crossorigin, computed from these exact pinned files.
    assert (
        'integrity="sha384-Hm6GpQcTNI5VqGgGS7lLxTGtEFcxu/kOVV0B7ozIZRu9blWVvigv5httJQZ2qZmY"'
        in r.text
    )
    assert (
        'integrity="sha384-Y7bC2PBKu8ujxtvo5+Z61OeGdSVRzFsYWBK4i5dnL/U6aFDTodk61qOUkTfInaxS"'
        in r.text
    )
    assert r.text.count('crossorigin="anonymous"') >= 2


def test_graph_page_ships_a_pinned_three_module_with_integrity(
    client: TestClient, auth_cookies
) -> None:
    """Restyle wave 2026-09-15 (graph-reference-aicodingdictionary.md):
    flat unlit node discs in 3D need direct `THREE.Sprite`/
    `THREE.CanvasTexture` constructors, but 3d-force-graph@1.80.0's UMD
    bundle never exposes a `THREE` global, and no three.js version in that
    library's own `three: >=0.179 <1` dependency range still ships a
    global/UMD build (checked directly on jsdelivr -- only ESM `build/
    three.module.js` exists any more). A pinned `type="module"` script
    with a real sha384 integrity hash is what actually loads and verifies
    it."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    assert 'src="https://cdn.jsdelivr.net/npm/three@0.184.0/build/three.module.min.js"' in r.text
    assert 'type="module"' in r.text
    assert (
        'integrity="sha384-35WaVqXNGvw49+RCXk67Ci9ebo55/5+CfirDbAVSG21KV8/7gO4xwIkhA5ZRcUHc"'
        in r.text
    )
    assert "@latest" not in r.text


def test_htmx_script_tag_carries_a_real_sri_hash(client: TestClient) -> None:
    """Review finding 9: `base.html`'s htmx script (loaded on every page)
    is the other unpinned-by-hash origin the review named."""
    r = client.get("/login")
    assert r.status_code == 200
    assert (
        'integrity="sha384-ujb1lZYygJmzgSwoxRggbCHcjc0rB2XoQrxeTUQyRjrOnlCoYta87iKBWq3EsdM2"'
        in r.text
    )
    assert 'crossorigin="anonymous"' in r.text


# --------------------------------------------------------------------
# Review finding 9 (second half): a real, per-request-nonce Content-
# Security-Policy, without breaking any page.
# --------------------------------------------------------------------


def test_csp_header_present_with_a_nonce_matching_the_pages_own_inline_scripts(
    client: TestClient, auth_cookies
) -> None:
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    csp = r.headers.get("content-security-policy")
    assert csp is not None
    assert "default-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "'unsafe-inline'" not in csp.split("style-src", 1)[0], (
        "script-src must never carry 'unsafe-inline' -- that would defeat the nonce entirely"
    )

    nonce_marker = "'nonce-"
    start = csp.index(nonce_marker) + len(nonce_marker)
    nonce = csp[start : csp.index("'", start)]
    assert nonce, "expected a real nonce value in the CSP header"
    # Every inline <script> on the page (base.html's two, chat.html's own)
    # must carry exactly this request's nonce -- proves the same value
    # threaded from the middleware through request.state into every
    # template that renders one, not just a hardcoded placeholder.
    assert r.text.count(f'nonce="{nonce}"') >= 3


def test_csp_allows_the_pinned_cdn_origins_for_scripts(client: TestClient, auth_cookies) -> None:
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    csp = r.headers["content-security-policy"]
    assert "https://cdnjs.cloudflare.com" in csp
    assert "https://cdn.jsdelivr.net" in csp


def test_how_it_works_has_no_content_security_policy_header(client: TestClient) -> None:
    """`/how-it-works` serves a standalone static file
    (`portolan.web.security`'s own module docstring) with its own inline
    scripts this middleware has no way to nonce -- excluded from the
    header entirely rather than shipping a policy that breaks the page."""
    client.cookies.clear()
    r = client.get("/how-it-works")
    assert r.status_code == 200
    assert "content-security-policy" not in {k.lower() for k in r.headers}


def test_authenticated_pages_carry_no_store_referrer_and_nosniff_headers(
    client: TestClient, auth_cookies
) -> None:
    """Bug hunt 2026-09-18, finding 10: `GET /` used to return the CSP
    header only -- `Cache-Control`, `Referrer-Policy`, and
    `X-Content-Type-Options` were all absent everywhere. Concretely: the
    external RSS link on `original.html` sends a `Referer` carrying
    internal slugs/locators to a third-party site on click, and a shared
    proxy could cache a page full of customer quotes."""
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    assert r.headers.get("cache-control") == "private, no-store"
    assert r.headers.get("referrer-policy") == "same-origin"
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_security_headers_apply_even_on_the_csp_exempt_how_it_works_page(
    client: TestClient,
) -> None:
    """Only the CSP itself is exempted on `/how-it-works` -- none of
    these three headers touches that page's own inline scripts or
    `data:` audio, so there's no reason to leave them off."""
    client.cookies.clear()
    r = client.get("/how-it-works")
    assert r.status_code == 200
    assert r.headers.get("referrer-policy") == "same-origin"
    assert r.headers.get("x-content-type-options") == "nosniff"
    # /how-it-works sets its own, deliberately cacheable Cache-Control
    # (test_how_it_works_is_cheap_on_repeat_requests) -- the security
    # middleware's `setdefault` must not clobber it.
    assert r.headers.get("cache-control") == "public, max-age=3600"


def test_original_raw_bytes_x_content_type_options_is_not_duplicated(
    client: TestClient, auth_cookies, call_source
) -> None:
    """`routes/originals.py`'s raw-bytes response already sets its own
    `X-Content-Type-Options` -- the security middleware's `setdefault`
    (not `append`) must not add a second copy of the header."""
    slug, _ = call_source
    r = client.get(f"/originals/{slug}/raw", cookies=auth_cookies)
    assert r.status_code == 200
    assert r.headers.get_list("x-content-type-options") == ["nosniff"]


def test_review_dismiss_button_has_no_inline_onclick_handler(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """The one inline event-handler attribute anywhere in the app (a
    nonce covers `<script>` elements, not inline `onclick=` attributes)
    -- moved to a `data-confirm` attribute plus an ordinary, nonce'd
    script so the CSP above doesn't have to carry `'unsafe-inline'` for
    scripts to keep this one dialog working."""
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('other', %s) RETURNING id",
            ('{"detail": "csp regression check"}',),
        )
    demo_conn.commit()

    r = client.get("/review", cookies=auth_cookies)
    assert r.status_code == 200
    assert "onclick=" not in r.text
    assert "data-confirm=" in r.text


def test_graph_page_has_an_error_message_element_wired_to_a_try_catch(
    client: TestClient, auth_cookies
) -> None:
    """Fix wave 2026-09-12: the root cause of "stuck on Loading the graph"
    was that the loading caption's own CSS made `.hidden = true` a no-op
    (var/graph-fix/repro.md) -- verified in a real headless-Chrome repro,
    the caption sat on top of a fully-rendered graph forever. The fix
    wraps the whole init in a try/catch that reports to a dedicated,
    visible error element (never the same element as the loading
    caption) naming the mechanism, with a link back to `/search`."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    assert 'id="graph-error"' in r.text
    assert 'id="graph-error-text"' in r.text
    assert "try {" in r.text
    assert "catch (err)" in r.text or "catch(err)" in r.text
    # the two literal, mechanism-naming failure messages the fix requires
    assert "The graph library did not load." in r.text
    assert "The graph data request failed with status " in r.text
    # the CSS bug's actual fix: display is scoped to :not([hidden]), so
    # the browser's native hidden behaviour and the author's display rule
    # can never fight each other again.
    assert ":not([hidden])" in r.text
    assert 'href="/search"' in r.text


def test_graph_redraw_itself_is_wrapped_in_try_catch_with_a_3d_to_2d_fallback(
    client: TestClient, auth_cookies
) -> None:
    """Review finding 5: the try/catch above only ever covered the
    *initial* `fetch()` -- `redraw()` is also called, unguarded, from the
    mode buttons, every legend chip, and the search debounce, and
    `draw3D`'s first line throws when the 3D library never loaded or the
    browser has no WebGL. Before the fix, any control clicked after the
    first successful load could throw into nothing: `container2d`/
    `container3d` were already hidden by that point, leaving an empty
    bordered box with no error and no loading caption. `redraw()` must
    wrap its own body in try/catch, report to the same error element, and
    fall back to 2D specifically when the 3D engine is what failed."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    redraw_start = text.index("function redraw() {")
    redraw_body = text[redraw_start : text.index("function setMode(mode)")]
    assert "try {" in redraw_body
    assert "catch (err)" in redraw_body
    assert "showError(" in redraw_body
    assert "setMode('2d')" in redraw_body


def test_graph_uses_one_engine_per_mode_and_does_not_copy_node_objects(
    client: TestClient, auth_cookies
) -> None:
    """Review finding 11: every filter/search-driven redraw used to
    dispose and reconstruct the live engine from scratch, and
    `currentGraphData()` copied every node object (`Object.assign({}, n)`)
    on top of that -- together, that discarded the x/y/vx/vy the
    simulation had already computed and the reader's zoom/pan on every
    legend click or keystroke. The fix: `currentGraphData()` hands back
    the real node objects (only links are still copied -- force-graph
    mutates a link's source/target into a node reference in place), and
    draw2D/draw3D reuse the live engine via `graphData()` when the mode
    hasn't changed, constructing a fresh one only on an actual switch."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text

    current_graph_data_start = text.index("function currentGraphData() {")
    current_graph_data_body = text[
        current_graph_data_start : text.index(
            "}", text.index("return { nodes", current_graph_data_start)
        )
    ]
    assert "keptNodes()" in current_graph_data_body
    assert "Object.assign({}, n)" not in current_graph_data_body, (
        "nodes must not be copied -- a copy loses the simulation's x/y/vx/vy"
    )
    assert "Object.assign({}, e)" in current_graph_data_body, (
        "links must still be copied fresh every call"
    )

    assert "var engineMode = null;" in text
    draw2d_start = text.index("function draw2D(built) {")
    draw2d_body = text[draw2d_start : text.index("function draw3D(built)")]
    assert "engine && engineMode === '2d'" in draw2d_body
    assert ".graphData({ nodes: built.nodes, links: built.links })" in draw2d_body

    draw3d_start = text.index("function draw3D(built) {")
    draw3d_body = text[draw3d_start : text.index("function emptyStateText")]
    assert "engine && engineMode === '3d'" in draw3d_body


def test_graph_json_reports_degree_per_node(client: TestClient, auth_cookies) -> None:
    r = client.get("/graph.json", cookies=auth_cookies)
    assert r.status_code == 200
    data = r.json()
    assert data["nodes"]
    assert all("degree" in node and isinstance(node["degree"], int) for node in data["nodes"])


def test_graph_json_reports_total_notes_and_truncation(client: TestClient, auth_cookies) -> None:
    r = client.get("/graph.json", cookies=auth_cookies)
    assert r.status_code == 200
    data = r.json()
    assert data["total_notes"] >= len(data["nodes"])
    assert data["truncated"] == (data["total_notes"] > len(data["nodes"]))
    assert data["node_cap"] == 2000


def test_graph_page_has_a_text_alternative_link_to_search(client: TestClient, auth_cookies) -> None:
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    assert "aria-label" in r.text  # the canvas carries an aria-label
    assert 'href="/search"' in r.text  # and a plain-text escape hatch


def test_graph_json_shape_is_unchanged_by_the_viewer_rewrite(
    client: TestClient, auth_cookies
) -> None:
    """Fix wave 2026-09-12 rewrote the viewer (`graph.html`'s inline
    script, force-graph/3d-force-graph) but not the data contract --
    `routes/graph.py` was not touched. Pin the exact top-level shape and
    one node's shape so a future change to either surface is deliberate."""
    r = client.get("/graph.json", cookies=auth_cookies)
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"nodes", "edges", "total_notes", "truncated", "node_cap"}
    assert data["nodes"], "expected the demo graph to return at least one node"
    assert set(data["nodes"][0].keys()) == {"id", "title", "type", "updated", "degree"}
    assert data["edges"], "expected the demo graph to return at least one edge"
    assert set(data["edges"][0].keys()) == {"source", "target", "kind"}
    assert isinstance(data["total_notes"], int)
    assert isinstance(data["truncated"], bool)
    assert data["node_cap"] == 2000


# --------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------


def test_coverage_requires_session(client: TestClient) -> None:
    r = client.get("/coverage", follow_redirects=False)
    assert r.status_code == 302


def test_coverage_lists_failed_source(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    with demo_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, extraction_status, s3_key)
            VALUES ('S-web-test-failed', 'call', 'failed', 'originals/call/S-web-test-failed')
            ON CONFLICT (slug) DO NOTHING
            """
        )

    r = client.get("/coverage", cookies=auth_cookies)
    assert r.status_code == 200
    assert "S-web-test-failed" in r.text


def test_coverage_days_out_of_range_returns_400_not_500(client: TestClient, auth_cookies) -> None:
    """Review finding F15, proven against a live app: `GET
    /coverage?days=99999999` returned a bare 500 (OverflowError building
    the timedelta)."""
    r = client.get("/coverage", params={"days": 99999999}, cookies=auth_cookies)
    assert r.status_code == 400


def test_coverage_negative_days_returns_400(client: TestClient, auth_cookies) -> None:
    r = client.get("/coverage", params={"days": -1}, cookies=auth_cookies)
    assert r.status_code == 400


# --------------------------------------------------------------------
# Review
# --------------------------------------------------------------------


def test_review_requires_session(client: TestClient) -> None:
    r = client.get("/review", follow_redirects=False)
    assert r.status_code == 302


def test_review_resolves_a_queue_row(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('other', %s) RETURNING id",
            ('{"note": "inserted for test_web.py"}',),
        )
        (row_id,) = cur.fetchone()

    r = client.get("/review", cookies=auth_cookies)
    assert str(row_id) in r.text or "inserted for test_web.py" in r.text

    r = client.post(f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False)
    assert r.status_code == 303

    status = _one(demo_conn, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0]
    assert status == "resolved"


def test_review_resolve_of_a_theme_resolution_row_confirms_it(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Nit 19 fix (review finding 19): resolving a `kind='theme_
    resolution'` row now stamps `theme_resolutions.confirmed_by`/
    `confirmed_at`, not just the queue row's own status -- the mirror of
    `test_review_dismiss_of_a_theme_resolution_row_reopens_the_theme`
    below."""
    from datetime import UTC, date, datetime

    from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
    from portolan.report import themes

    slug = "theme-webreview-confirm-fixture"
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.THEME,
        title="Web review confirm fixture",
        description="A theme note built only to test /review resolve.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Test fixture.",
    )
    write_note(demo_conn, note, [], changed_by="test", reason="web review fixture")

    with demo_conn.cursor() as cur:
        cur.execute("SELECT id FROM claims LIMIT 1")
        (any_claim_id,) = cur.fetchone()

    themes._mark_theme_resolved(
        demo_conn,
        slug,
        resolution_claim_id=any_claim_id,
        resolved_week=date(2026, 8, 3),
        reason="test fixture resolution",
    )
    active = themes._get_theme_resolution(demo_conn, slug)
    assert active is not None
    assert active.confirmed_by is None

    with demo_conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM review_queue WHERE payload->>'theme_slug' = %s "
            "ORDER BY id DESC LIMIT 1",
            (slug,),
        )
        (row_id,) = cur.fetchone()

    r = client.post(f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False)
    assert r.status_code == 303

    active = themes._get_theme_resolution(demo_conn, slug)
    assert active is not None
    assert active.confirmed_by == "Test Editor"  # the auth_cookies fixture's own login editor
    assert active.confirmed_at is not None

    status = _one(demo_conn, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0]
    assert status == "resolved"


def test_review_resolve_of_a_proposal_that_survived_a_merge_confirms_it(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Blocker 2 (review-theme-wave-5c.md finding 2): a merge used to
    repoint the `theme_resolutions` row but never the `review_queue` row
    `_queue_resolution_for_confirmation` wrote for it -- so a proposal
    that survived a merge (the absorbed theme's own open proposal, which
    outranks the keep side's empty history) became un-confirmable and
    un-rejectable forever through `/review`: both buttons landed on
    `?notice=superseded`, which was a lie for a proposal that was still
    very much alive under the surviving slug. This is the end-to-end
    route-level half of the fix (test gap 6); the direct
    `confirm_theme_resolution` half is covered by
    `test_report_themes.py::test_merge_never_lets_a_silence_row_
    supersede_a_pending_proposal`."""
    from datetime import UTC, datetime

    from test_report_themes import _add_member, _insert_raw_claim

    from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
    from portolan.report import themes

    now = datetime.now(UTC)
    keep_slug = "theme-webreview-mergesurvive-keep"
    absorb_slug = "theme-webreview-mergesurvive-absorb"
    for slug in (keep_slug, absorb_slug):
        write_note(
            demo_conn,
            NoteRow(
                slug=slug,
                type=NoteType.THEME,
                title=slug,
                description="A theme note built only to test /review after a merge.",
                status=NoteStatus.LIVING,
                created=now,
                updated=now,
                synthesis="Test fixture.",
            ),
            [],
            changed_by="test",
            reason="web review merge-survives fixture",
        )

    with demo_conn.cursor() as cur:
        cur.execute("SELECT slug FROM sources LIMIT 1")
        (any_source,) = cur.fetchone()

    keep_member = _insert_raw_claim(
        demo_conn,
        keep_slug,
        "risk",
        "[[st-aldric-health]] flagged a webreviewmergesurvive alpha risk.",
        any_source,
        "webreviewmergesurvive#a",
        date(2026, 8, 4),
    )
    _add_member(demo_conn, keep_slug, keep_member, date(2026, 8, 4))
    absorb_member = _insert_raw_claim(
        demo_conn,
        absorb_slug,
        "risk",
        "[[northlake-health-system]] flagged a webreviewmergesurvive beta risk.",
        any_source,
        "webreviewmergesurvive#b",
        date(2026, 8, 5),
    )
    _add_member(demo_conn, absorb_slug, absorb_member, date(2026, 8, 5))

    # An open proposal on the ABSORBED side -- the ordinary merge shape
    # the review measured (only the absorbed theme carries a live
    # proposal); the keep side has no active resolution at all, so the
    # absorbed side's proposal wins outright and must survive the merge.
    absorb_praise = _insert_raw_claim(
        demo_conn,
        absorb_slug,
        "praise",
        "[[northlake-health-system]] said the webreviewmergesurvive situation is fixed.",
        any_source,
        "webreviewmergesurvive#c",
        date(2026, 8, 20),
    )
    themes._mark_theme_resolved(
        demo_conn,
        absorb_slug,
        resolution_claim_id=absorb_praise,
        resolved_week=date(2026, 8, 20),
        reason="absorb-side proposal, unconfirmed",
        status="proposed",
    )

    themes._merge_themes(demo_conn, keep_slug, absorb_slug, reason="test merge for /review route")

    survivor = themes._get_theme_resolution(demo_conn, keep_slug)
    assert survivor is not None
    assert survivor.resolution_claim_id == absorb_praise
    assert survivor.status == "proposed"

    with demo_conn.cursor() as cur:
        cur.execute(
            "SELECT id, payload->>'theme_slug' FROM review_queue "
            "WHERE kind = 'theme_resolution' AND status = 'open' "
            "AND payload->>'resolution_claim_id' = %s",
            (str(absorb_praise),),
        )
        row_id, queue_theme_slug = cur.fetchone()
    # Repointed to the surviving slug, not left on the absorbed one.
    assert queue_theme_slug == keep_slug

    r = client.post(f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False)
    assert r.status_code == 303
    # The whole point: NOT `?notice=superseded` for a proposal that is
    # still alive.
    assert r.headers["location"] == "/review?notice=confirmed"

    confirmed = themes._get_theme_resolution(demo_conn, keep_slug)
    assert confirmed is not None
    assert confirmed.status == "confirmed"
    assert confirmed.confirmed_by == "Test Editor"

    status = _one(demo_conn, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0]
    assert status == "resolved"


def test_review_resolve_of_an_already_superseded_theme_resolution_row_shows_a_notice(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Should-fix 8 / test gap 7 (`01-spec/reviews/review-theme-fix-wave-
    3.md`): the confirm side of the exact defect should-fix 11 of the
    PRIOR wave fixed on dismiss -- a confirm whose `resolution_claim_id`
    no longer matches the theme's CURRENT active resolution (superseded
    some other way before the human's click reached the server) is a
    no-op on the theme, but used to redirect identically to a confirm
    that actually took effect. Mirrors `test_review_dismiss_of_an_
    already_superseded_theme_resolution_row_shows_a_notice` exactly, one
    route over."""
    from datetime import UTC, date, datetime

    from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
    from portolan.report import themes

    slug = "theme-webreview-stale-confirm-fixture"
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.THEME,
        title="Web review stale confirm fixture",
        description="A theme note built only to test a stale /review resolve.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Test fixture.",
    )
    write_note(demo_conn, note, [], changed_by="test", reason="web review fixture")

    with demo_conn.cursor() as cur:
        cur.execute("SELECT id FROM claims LIMIT 1")
        (any_claim_id,) = cur.fetchone()

    themes._mark_theme_resolved(
        demo_conn,
        slug,
        resolution_claim_id=any_claim_id,
        resolved_week=date(2026, 8, 3),
        reason="test fixture resolution",
    )
    with demo_conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM review_queue WHERE payload->>'theme_slug' = %s "
            "ORDER BY id DESC LIMIT 1",
            (slug,),
        )
        (row_id,) = cur.fetchone()

    # Supersede it some other way BEFORE the human's confirm click
    # reaches the server -- the same race the review's finding 11 named
    # for dismiss, applied to the mirror route.
    active = themes._get_theme_resolution(demo_conn, slug)
    assert themes.reject_theme_resolution(
        demo_conn,
        theme_slug=slug,
        resolution_claim_id=active.resolution_claim_id,
        changed_by="someone-else",
    )
    assert themes._get_theme_resolution(demo_conn, slug) is None

    r = client.post(f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/review?notice=superseded"

    status = _one(demo_conn, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0]
    assert status == "resolved"  # still a human's final word on the ROW

    page = client.get(r.headers["location"], cookies=auth_cookies)
    assert "already superseded" in page.text


def test_review_resolve_of_a_malformed_theme_resolution_row_shows_a_notice(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Test gap 7 (`01-spec/reviews/review-theme-fix-wave-3.md`): a
    `theme_resolution` queue row whose payload is missing `theme_slug`/
    `resolution_claim_id` used to be silently indistinguishable from a
    stale-but-genuine reject/confirm -- now gets its own distinct
    `notice=malformed`, on both routes (dismiss already had this; this
    pins the confirm side too)."""
    with demo_conn.cursor() as cur:
        cur.execute("SELECT slug FROM sources LIMIT 1")
        (any_source,) = cur.fetchone()
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug, status) "
            "VALUES ('theme_resolution', '{}'::jsonb, %s, 'open') RETURNING id",
            (any_source,),
        )
        (row_id,) = cur.fetchone()
    demo_conn.commit()

    r = client.post(f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/review?notice=malformed"

    status = _one(demo_conn, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0]
    assert status == "resolved"

    page = client.get(r.headers["location"], cookies=auth_cookies)
    assert "missing the theme/claim" in page.text


def test_review_dismiss_of_a_theme_resolution_row_reopens_the_theme(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Should-fix 4 / test gap 4 (review finding 4): dismissing a
    `kind='theme_resolution'` review-queue row must actually reopen the
    theme (`theme_resolutions.reopened_week`), not just flip the queue
    row's own status -- the gap the review's finding 4 named: "a human
    who opens `/review`, sees a wrong resolution, and clicks dismiss has
    changed the queue row's status and nothing about the theme.\""""
    from datetime import UTC, date, datetime

    from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
    from portolan.report import themes

    slug = "theme-webreview-dismiss-fixture"
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.THEME,
        title="Web review dismiss fixture",
        description="A theme note built only to test /review dismiss.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Test fixture.",
    )
    write_note(demo_conn, note, [], changed_by="test", reason="web review fixture")

    with demo_conn.cursor() as cur:
        cur.execute("SELECT id FROM claims LIMIT 1")
        (any_claim_id,) = cur.fetchone()

    # _mark_theme_resolved writes both the theme_resolutions row AND its
    # own pending-confirmation review_queue row (`_queue_resolution_for_
    # confirmation`) -- exactly the shape a real weekly run produces.
    themes._mark_theme_resolved(
        demo_conn,
        slug,
        resolution_claim_id=any_claim_id,
        resolved_week=date(2026, 8, 3),
        reason="test fixture resolution",
    )
    assert themes._get_theme_resolution(demo_conn, slug) is not None

    with demo_conn.cursor() as cur:
        cur.execute(
            "SELECT id, kind FROM review_queue WHERE payload->>'theme_slug' = %s "
            "ORDER BY id DESC LIMIT 1",
            (slug,),
        )
        row_id, kind = cur.fetchone()
    assert kind == "theme_resolution"

    r = client.post(f"/review/{row_id}/dismiss", cookies=auth_cookies, follow_redirects=False)
    assert r.status_code == 303
    # Should-fix 7 (review-theme-wave-5-judge.md finding 7, "result
    # shown"): a rejection that actually took effect now redirects with
    # `?notice=rejected` -- a human clicking Reject gets told what
    # happened, not just a bare redirect. Contrast the stale-dismiss
    # test below, which still redirects with `?notice=superseded`.
    assert r.headers["location"] == "/review?notice=rejected"

    # The theme is un-resolved -- no longer an ACTIVE resolution ...
    assert themes._get_theme_resolution(demo_conn, slug) is None
    # ... but the row itself survives, marked reopened, not deleted.
    history = themes._theme_resolutions_history(demo_conn, slug)
    assert len(history) == 1
    assert history[0].reopened_week is not None

    status = _one(demo_conn, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0]
    assert status == "dismissed"


def test_review_page_renders_a_theme_resolution_proposal_as_a_proposal(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Should-fix 7 (review-theme-wave-5-judge.md finding 7) and test gap
    7 ("no tests/test_web.py coverage of a theme_resolution queue row --
    not its rendering, not the evidence being visible, not the two
    buttons' behaviour on one"): the page shows the theme's real NAME,
    the quoted EVIDENCE, the resolving claim's ACCOUNT and DATE, and the
    two proposal-specific button labels -- never the generic six-row
    payload dump every other row kind falls through to, and never the
    judge's own free-text `reason`.

    CI regression (2026-09-14 CI red): this test's own claim `INSERT`
    used to hand a bare `date(2026, 8, 12)` for `claims.recorded`
    (`timestamptz`), leaving Postgres to cast date -> timestamptz by
    stamping midnight in whatever the *session's* `TimeZone` happened to
    be -- correct by coincidence on a dev machine whose Postgres defaults
    to `America/Phoenix` (this build's own `PORTOLAN_TIMEZONE` default),
    wrong in CI (session `TimeZone` `UTC`, runner `TZ=UTC`): midnight
    2026-08-12 UTC reads back as 2026-08-11 once `clock.world_date`
    converts it to Phoenix for display. `portolan.merge.util.recorded_dt`
    already fixed this exact class of bug for the real ingest path
    (review-report-audit-consolidate.md finding 1) by anchoring midnight
    explicitly *in the deployment's own zone* instead of leaving the cast
    to the session -- this test now does the same, and forces the
    session to `UTC` first (`SET TIME ZONE`, reset after) so it fails
    before that fix and passes after, on any machine, not just in CI."""
    from datetime import UTC, date, datetime, time
    from zoneinfo import ZoneInfo

    from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
    from portolan.report import themes
    from portolan.util import clock

    account_slug = _one(demo_conn, "SELECT slug FROM notes WHERE type = 'account' LIMIT 1")[0]

    theme_slug = "theme-webreview-render-fixture"
    now = datetime.now(UTC)
    theme_note = NoteRow(
        slug=theme_slug,
        type=NoteType.THEME,
        title="Web Review Render Fixture Theme",
        description="A theme note built only to test /review's proposal rendering.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Test fixture.",
    )
    write_note(demo_conn, theme_note, [], changed_by="test", reason="web review render fixture")

    signal_slug = "signal-webreview-render-fixture"
    signal_note = NoteRow(
        slug=signal_slug,
        type=NoteType.SIGNAL,
        title=signal_slug,
        description="A signal note built only to test /review's proposal rendering.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
    )
    write_note(demo_conn, signal_note, [], changed_by="test", reason="web review render fixture")

    # Force the session's ambient TimeZone to UTC (CI's own session
    # TimeZone) for the rest of this test, so the `recorded` write below
    # is proven to round-trip correctly through `clock.world_date`
    # regardless of the reading/writing session's own zone -- not just on
    # a dev machine whose Postgres session defaults to
    # `PORTOLAN_TIMEZONE`'s own default by coincidence. `demo_conn` is
    # session-scoped (shared with every other test in this module), so
    # this always resets the zone back in a `finally`.
    with demo_conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'UTC'")
    try:
        recorded_at = datetime.combine(
            date(2026, 8, 12), time.min, tzinfo=ZoneInfo(clock.timezone_name())
        )
        with demo_conn.cursor() as cur:
            cur.execute("SELECT slug FROM sources LIMIT 1")
            (any_source,) = cur.fetchone()
            cur.execute(
                "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
                "recorded) VALUES (%s, 'praise', %s, %s, 'webreview#a', %s) RETURNING id",
                (
                    signal_slug,
                    f"[[{account_slug}]] said the webreviewrenderfixture problem is fixed now.",
                    any_source,
                    recorded_at,
                ),
            )
            (claim_id,) = cur.fetchone()
        demo_conn.commit()

        themes._mark_theme_resolved(
            demo_conn,
            theme_slug,
            resolution_claim_id=claim_id,
            resolved_week=date(2026, 8, 10),
            reason="the judge's own internal reason text -- must never render on the page",
            status="proposed",
            evidence="the problem is fixed now",
        )

        r = client.get("/review", cookies=auth_cookies)
    finally:
        with demo_conn.cursor() as cur:
            cur.execute("RESET TIME ZONE")

    assert r.status_code == 200
    assert "Web Review Render Fixture Theme" in r.text  # the theme's NAME, not just its slug
    assert "the problem is fixed now" in r.text  # the quoted evidence
    # Nit 11 (review-theme-wave-5b.md finding 11, last bullet): a bare
    # `account_slug in r.text` passes even for the wrong reason -- this
    # page also renders an entity picker listing every registry entity
    # for `alias` rows, which would contain the same slug regardless of
    # whether the proposal block itself rendered anything. Pin the actual
    # link the proposal block renders (`review.html`'s own `<a href="/
    # notes/{{ row.proposal.account_slug }}">`) instead.
    assert f'<a href="/notes/{account_slug}">' in r.text  # the resolving claim's account, linked
    assert "2026-08-12" in r.text  # the resolving claim's own date
    assert "Confirm resolution" in r.text
    assert "Reject" in r.text
    assert "must never render on the page" not in r.text  # the judge's own reason -- never shown


def test_review_dismiss_of_an_already_superseded_theme_resolution_row_shows_a_notice(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Should-fix 11 (review finding 11): a dismiss whose `resolution_
    claim_id` no longer matches the theme's CURRENT active resolution
    (already superseded some other way) is a no-op on the theme -- the
    queue row still lands `dismissed` (a human's final word on that row),
    but the redirect now carries `?notice=superseded` instead of looking
    identical to a dismiss that actually took effect."""
    from datetime import UTC, date, datetime

    from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
    from portolan.report import themes

    slug = "theme-webreview-stale-dismiss-fixture"
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.THEME,
        title="Web review stale dismiss fixture",
        description="A theme note built only to test a stale /review dismiss.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Test fixture.",
    )
    write_note(demo_conn, note, [], changed_by="test", reason="web review fixture")

    with demo_conn.cursor() as cur:
        cur.execute("SELECT id FROM claims LIMIT 1")
        (any_claim_id,) = cur.fetchone()

    themes._mark_theme_resolved(
        demo_conn,
        slug,
        resolution_claim_id=any_claim_id,
        resolved_week=date(2026, 8, 3),
        reason="test fixture resolution",
    )
    with demo_conn.cursor() as cur:
        cur.execute(
            "SELECT id, payload FROM review_queue WHERE payload->>'theme_slug' = %s "
            "ORDER BY id DESC LIMIT 1",
            (slug,),
        )
        row_id, _payload = cur.fetchone()

    # Supersede it some other way BEFORE the human's dismiss click
    # reaches the server -- the same race the review's finding names.
    active = themes._get_theme_resolution(demo_conn, slug)
    assert themes.reject_theme_resolution(
        demo_conn,
        theme_slug=slug,
        resolution_claim_id=active.resolution_claim_id,
        changed_by="someone-else",
    )
    assert themes._get_theme_resolution(demo_conn, slug) is None

    r = client.post(f"/review/{row_id}/dismiss", cookies=auth_cookies, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/review?notice=superseded"

    status = _one(demo_conn, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0]
    assert status == "dismissed"  # still a human's final word on the ROW

    # The notice renders in plain language on the page itself.
    page = client.get(r.headers["location"], cookies=auth_cookies)
    assert "already superseded" in page.text


def test_review_add_alias_for_unknown_entity(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('unknown_entity', %s) RETURNING id",
            ('{"mention": "test-mention-web", "context": {}}',),
        )
        (row_id,) = cur.fetchone()
        cur.execute("SELECT crm_id FROM registry_entities WHERE crm_id IS NOT NULL LIMIT 1")
        entity_row = cur.fetchone()
    assert entity_row is not None
    (entity_id,) = entity_row

    r = client.post(
        f"/review/{row_id}/add-alias",
        data={"entity_id": entity_id},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r.status_code == 303

    alias_row = _one(
        demo_conn, "SELECT alias FROM registry_aliases WHERE alias = %s", ("test-mention-web",)
    )
    assert alias_row is not None
    status = _one(demo_conn, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0]
    assert status == "resolved"


def test_review_add_alias_with_a_bogus_entity_id_returns_400_not_500(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Review finding F18: `POST /review/1/add-alias` with a bogus
    `entity_id` returned a bare 500 -- `add_alias`'s `ValueError` for an
    unknown entity escaped uncaught."""
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('unknown_entity', %s) RETURNING id",
            ('{"mention": "test-mention-bogus", "context": {}}',),
        )
        (row_id,) = cur.fetchone()

    r = client.post(
        f"/review/{row_id}/add-alias",
        data={"entity_id": "DOES-NOT-EXIST"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r.status_code == 400


# --------------------------------------------------------------------
# Search / reports
# --------------------------------------------------------------------


def test_search_redirects_to_atlas(client: TestClient) -> None:
    """Atlas wave 2026-09-18: `/search` is now a thin, permanent redirect
    to `/atlas` -- old bookmarks/links keep working. No session needed to
    be redirected (the target, `/atlas`, is what actually requires one --
    the same "302 to /login" a logged-out reader would already get by
    following the redirect one hop further)."""
    r = client.get("/search", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/atlas"


def test_search_redirect_preserves_query(client: TestClient) -> None:
    r = client.get("/search", params={"q": "renewal"}, follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/atlas?q=renewal"


def test_search_redirect_then_atlas_ok(client: TestClient, auth_cookies) -> None:
    r = client.get("/search", params={"q": "renewal"}, cookies=auth_cookies)
    assert r.status_code == 200
    assert r.request.url.path == "/atlas"


def test_search_query_over_the_limit_returns_400_not_500(client: TestClient, auth_cookies) -> None:
    """Review finding 19: `/search?q=` went straight to the embedder and
    into a regex alternation in `snippets.highlight_terms` with no bound
    at all."""
    r = client.get("/search", params={"q": "x" * 501}, cookies=auth_cookies)
    assert r.status_code == 400


def test_notes_query_over_the_limit_returns_400_not_500(client: TestClient, auth_cookies) -> None:
    r = client.get("/notes", params={"q": "x" * 201}, cookies=auth_cookies)
    assert r.status_code == 400


def test_edit_form_after_text_over_the_limit_returns_400_not_500(
    client: TestClient, auth_cookies, any_claim
) -> None:
    note_slug, *_ = any_claim
    r = client.post(
        f"/notes/{note_slug}/edit",
        data={"after_text": "x" * 20_001},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r.status_code == 400


def test_edit_company_context_over_its_own_2500_char_cap_returns_400(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Owner decision 2026-09-17, "Company context page": the context
    note's own cap (`portolan.context.COMPANY_CONTEXT_BODY_CHAR_CAP`) is
    tighter than the generic 20,000-char edit-form limit above, and is
    enforced in the route itself (`portolan.web.routes.notes.note_edit_
    submit`), not just the lint."""
    from portolan.context import COMPANY_CONTEXT_BODY_CHAR_CAP, seed_company_context

    seed_company_context(demo_conn)

    r = client.post(
        "/notes/company-context/edit",
        data={"after_text": "x" * (COMPANY_CONTEXT_BODY_CHAR_CAP + 1), "reason": "test"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r.status_code == 400

    r_ok = client.post(
        "/notes/company-context/edit",
        data={"after_text": "x" * COMPANY_CONTEXT_BODY_CHAR_CAP, "reason": "test"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r_ok.status_code == 303


def test_conversations_create_has_a_per_owner_cap(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Review finding 19: `POST /conversations` had no cap at all."""
    from portolan.web.conversations import create_conversation

    owner = "Cap Tester"
    with demo_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM conversations WHERE owner = %s", (owner,))
        (existing,) = cur.fetchone()
    from portolan.web.routes.chat import MAX_CONVERSATIONS_PER_OWNER

    for _ in range(MAX_CONVERSATIONS_PER_OWNER - existing):
        create_conversation(demo_conn, owner=owner)

    login = client.post(
        "/login", data={"org_key": ORG_KEY, "editor": owner}, follow_redirects=False
    )
    r = client.post("/conversations", cookies=login.cookies, follow_redirects=False)
    assert r.status_code == 400


def test_reports_requires_session(client: TestClient) -> None:
    r = client.get("/reports", follow_redirects=False)
    assert r.status_code == 302


def test_reports_list_ok(client: TestClient, auth_cookies) -> None:
    r = client.get("/reports", cookies=auth_cookies)
    assert r.status_code == 200


def test_reports_detail_404s_for_a_note_that_is_not_a_report(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Review finding F18: `routes/reports.py` didn't filter `type =
    'report'`, so `/reports/{a signal note}` returned 200 under a
    "Reports" breadcrumb."""
    row = _one(demo_conn, "SELECT slug FROM notes WHERE type != 'report' LIMIT 1")
    assert row is not None
    (slug,) = row
    r = client.get(f"/reports/{slug}", cookies=auth_cookies)
    assert r.status_code == 404


# --------------------------------------------------------------------
# Every page returns 200 with the session, 302/401 without
# --------------------------------------------------------------------

_GET_PAGES = ["/", "/search", "/graph", "/reports", "/coverage", "/review", "/notes"]


@pytest.mark.parametrize("path", _GET_PAGES)
def test_page_200_with_session_302_without(client: TestClient, auth_cookies, path: str) -> None:
    with_session = client.get(path, cookies=auth_cookies)
    assert with_session.status_code == 200, f"{path} did not return 200 with a session"

    # `client` is a stateful session (its cookie jar persists across
    # requests) and `auth_cookies` already logged in on this same client,
    # so the "without a session" check needs to explicitly drop that
    # cookie rather than just omitting `cookies=` from the call.
    client.cookies.clear()
    without_session = client.get(path, follow_redirects=False)
    assert without_session.status_code == 302, f"{path} did not redirect without a session"


# --------------------------------------------------------------------
# UI wave 2 (01-spec/reviews/ux-eval-2026-09-12.md): the design system and
# its ranked punch list, applied on top of wave 1's shell/theme
# toggle/graph modes/chat sessions (left untouched by everything below).
# --------------------------------------------------------------------

_WAVE2_PAGES = [
    "/",
    "/search",
    "/graph",
    "/reports",
    "/coverage",
    "/review",
    "/notes",
    "/login",
]


def _all_font_sizes_in_inline_styles(html: str) -> list[int]:
    return [int(n) for n in re.findall(r'style="[^"]*font-size:\s*(\d+)px', html)]


@pytest.mark.parametrize("path", _WAVE2_PAGES)
def test_no_font_size_below_13px_in_inline_styles(
    client: TestClient, auth_cookies, any_claim, path: str
) -> None:
    """Punch item 7 (dates/source lines "must never be decorative grey")
    and the wave's own verification instruction: measure this in code, not
    by eye. Scoped to inline `style="..."` attributes -- a `<style>` block
    rule can still legitimately sit at exactly 12px (badges, table `th`,
    the footer), but nothing rendered inline may ever drop below 13px."""
    r = client.get(path, cookies=auth_cookies)
    assert r.status_code == 200
    sizes = _all_font_sizes_in_inline_styles(r.text)
    assert all(n >= 13 for n in sizes), f"{path} has an inline font-size below 13px: {sizes}"


def test_no_font_size_below_13px_on_a_note_page(
    client: TestClient, auth_cookies, any_claim
) -> None:
    note_slug, *_ = any_claim
    r = client.get(f"/notes/{note_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    sizes = _all_font_sizes_in_inline_styles(r.text)
    assert all(n >= 13 for n in sizes), f"note page has an inline font-size below 13px: {sizes}"


def test_no_font_size_below_13px_on_a_source_and_original_page(
    client: TestClient, auth_cookies, call_source
) -> None:
    slug, locator = call_source
    for r in (
        client.get(f"/sources/{slug}", cookies=auth_cookies),
        client.get(f"/originals/{slug}", params={"at": locator}, cookies=auth_cookies),
    ):
        assert r.status_code == 200
        sizes = _all_font_sizes_in_inline_styles(r.text)
        assert all(n >= 13 for n in sizes), f"has an inline font-size below 13px: {sizes}"


# --------------------------------------------------------------------
# Punch item 1 and 3, verified again here (wave 1 already fixed both --
# this wave's job was to confirm and move on, not re-fix).
# --------------------------------------------------------------------


def test_wave1_fixes_still_hold_brand_css_and_accent_reservation(
    client: TestClient, auth_cookies
) -> None:
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    style_start = r.text.index("<style>")
    style_end = r.text.index("</style>", style_start)
    style_block = r.text[style_start:style_end]
    assert "&#34;" not in style_block
    assert '--pl-display: "Avenir Next"' in style_block
    assert "a { color: var(--pl-ink)" in style_block


# --------------------------------------------------------------------
# Punch item 2: search snippets never show raw frontmatter.
# --------------------------------------------------------------------


def test_search_snippet_strips_frontmatter_and_falls_back_to_description() -> None:
    from portolan.web.snippets import search_snippet

    chunk = '---\ntitle: "Complaint"\ndescription: "d"\nstatus: living\n---\n'
    out = str(search_snippet(chunk, "fallback description text", "renewal"))
    assert "description:" not in out
    assert "---" not in out
    assert out.strip() == "fallback description text"

    prose_chunk = chunk + "This is the real body paragraph about renewal terms."
    out2 = str(search_snippet(prose_chunk, "unused fallback", "renewal"))
    assert "description:" not in out2
    assert "<mark>renewal</mark>" in out2


def test_search_snippet_strips_claim_markdown_furniture() -> None:
    """A chunk drawn from a note's "## Claims"/"## Related" section
    carries the same claim bullet markdown `portolan.notes.render` writes
    into `body_md` -- a leading `- `, `**bold**`, a `^[source @ locator]`
    citation marker, and a `[[slug]]` wikilink. None of that belongs in a
    plain-prose search snippet."""
    from portolan.web.snippets import search_snippet

    claim_chunk = (
        "- **2026-04-13** | decision | Ashgrove Health's contract renews on 2027-03-31. "
        "([[halvardcare-service-contract]] [[ashgrove-health]]) ^[S-call-003 @ 00:04:15]"
    )
    out = str(search_snippet(claim_chunk, None, "renews"))
    assert "[[" not in out
    assert "^[" not in out
    assert "**" not in out
    assert out.startswith("-") is False
    assert "<mark>renews</mark>" in out


def test_atlas_search_pane_snippet_never_shows_raw_frontmatter(
    client: TestClient, auth_cookies, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX eval 2026-09-12, punch item 2: `hit.best_chunk[:220]` used to
    print raw YAML straight into the card a rep uses to pick a hit. That
    behaviour now lives on `/atlas` (Atlas wave 2026-09-18, "the old
    /search page's function lives here now") -- the raw-explain()
    disclosure this used to also check is gone along with the rest of
    the old plain search page (`explain()` is still used elsewhere, by
    chat); only the note-card snippet's own "never raw frontmatter"
    guarantee is re-asserted here, against its new home."""
    from portolan.search.query import SearchHit, SearchResult
    from portolan.web.routes import atlas as atlas_route

    hostile_chunk = (
        '---\ntitle: "Complaint — Meadowbrook"\ndescription: "d"\nstatus: living\n'
        "source:\n  - S-ticket-03\n---\n"
        "This is the real body paragraph mentioning apple and banana."
    )
    hit = SearchHit(
        note_slug="does-not-exist-in-the-demo-graph",
        title="Synthetic hit",
        score=1.0,
        via=["fts"],
        best_chunk=hostile_chunk,
        heading=None,
        updated=date(2026, 9, 1),
        source_slugs=[],
    )
    fake_result = SearchResult(hits=[hit], control_ok=True, query_terms_unmatched=[])
    monkeypatch.setattr(atlas_route, "run_search", lambda *a, **k: fake_result)

    r = client.get("/atlas", params={"q": "apple"}, cookies=auth_cookies)
    assert r.status_code == 200
    match = re.search(r'<p class="note-card-body[^"]*">(.*?)</p>', r.text, re.DOTALL)
    assert match is not None, "search hit card body not found"
    card_body = match.group(1)
    assert "description:" not in card_body
    assert "---" not in card_body
    assert "<mark>apple</mark>" in card_body


# --------------------------------------------------------------------
# Atlas (build instructions, 2026-09-18): the Obsidian-style note
# browser that replaces the old plain /search page.
# --------------------------------------------------------------------


def test_atlas_requires_session(client: TestClient) -> None:
    r = client.get("/atlas", follow_redirects=False)
    assert r.status_code == 302


def test_atlas_home_shows_folder_counts_and_empty_state(client: TestClient, auth_cookies) -> None:
    """No note chosen, no query: the empty state's "a short explainer and
    the counts per folder" (build instructions)."""
    r = client.get("/atlas", cookies=auth_cookies)
    assert r.status_code == 200
    assert "Browse the graph" in r.text
    # The demo graph always has account notes; its folder chip names the
    # plain-word label, not the raw `account` type string.
    assert "Accounts" in r.text
    assert 'href="/notes?type=account"' in r.text


def test_atlas_tree_folder_labels_are_left_aligned_not_centred(
    client: TestClient, auth_cookies
) -> None:
    """Demo-polish fix (app checklist 2026-09-18, P0 item 5): the folder
    row (`_atlas_panel.html`'s `<summary>`) has THREE flex children -- the
    arrow glyph (a `::before` pseudo-element on `summary` itself, so it
    counts as the first flex item), the label `<span>`, and the count
    `<span>`. The old rule, `justify-content: space-between`, spread all
    three evenly across the row, which put the label -- the middle child
    -- dead centre instead of left-aligned right after the arrow, the
    "centred tree label" a design-partner screenshot caught. Proven at
    the CSS level (not a layout engine this test suite has) -- the fixed
    rule keeps everything flush left except the count, pushed right by
    its own `margin-left: auto`."""
    r = client.get("/atlas", cookies=auth_cookies)
    assert r.status_code == 200
    assert "justify-content: flex-start" in r.text
    assert "margin-left: auto" in r.text
    # The old, bug-causing rule must be gone from the folder/subfolder
    # summary selector specifically (not merely "not present anywhere" --
    # `space-between` is a legitimate value elsewhere on this page).
    folder_summary_rule = re.search(
        r"\.atlas-folder > summary, \.atlas-subfolder > summary \{[^}]*\}", r.text
    )
    assert folder_summary_rule is not None
    assert "space-between" not in folder_summary_rule.group(0)


def test_atlas_query_over_the_limit_returns_400_not_500(client: TestClient, auth_cookies) -> None:
    r = client.get("/atlas", params={"q": "x" * 501}, cookies=auth_cookies)
    assert r.status_code == 400


def test_atlas_note_page_404_for_missing_note(client: TestClient, auth_cookies) -> None:
    r = client.get("/atlas/no-such-note-at-all", cookies=auth_cookies)
    assert r.status_code == 404


def test_atlas_note_page_renders_breadcrumb_and_properties(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Breadcrumb (Folder / slug) and a Properties block with every
    frontmatter field as label/value rows -- type and status as chips,
    dates as dates, source as links (build instructions)."""
    note_slug, _claim_id, source_slug, _locator = any_claim
    row = _one(demo_conn, "SELECT type, version FROM notes WHERE slug = %s", (note_slug,))
    assert row is not None
    note_type, version = row

    r = client.get(f"/atlas/{note_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    # Breadcrumb: Atlas / <plain-word folder> / <slug>.
    assert f'href="/notes?type={note_type}"' in r.text
    assert f'<span class="mono">{note_slug}</span>' in r.text
    # Properties: the type badge, the status badge, created/updated dates,
    # the version number, and a source link citing this claim's own
    # source (every claim's source is a `provenance` link off the note).
    assert "Properties" in r.text
    assert f'<span class="badge type-{note_type}">{note_type}</span>' in r.text
    assert "Created" in r.text and "Updated" in r.text
    assert f">{version}</dd>" in r.text
    assert f'href="/originals/{source_slug}"' in r.text


def test_atlas_note_page_includes_edit_and_download_actions(
    client: TestClient, auth_cookies, any_claim
) -> None:
    note_slug, *_ = any_claim
    r = client.get(f"/atlas/{note_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert f'href="/notes/{note_slug}/edit"' in r.text
    assert f'href="/notes/{note_slug}.md"' in r.text


def test_atlas_note_page_renders_edges_section_both_directions(
    client: TestClient, auth_cookies, note_with_links_both_directions
) -> None:
    """Edges: outgoing links grouped by relation kind, and backlinks
    ("N notes link here"), every entry linking within Atlas
    (`/atlas/<slug>`) so the tree stays (build instructions)."""
    slug, outgoing_slug, incoming_slug = note_with_links_both_directions
    r = client.get(f"/atlas/{slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert "Edges" in r.text
    assert f'href="/atlas/{outgoing_slug}"' in r.text
    assert f'href="/atlas/{incoming_slug}"' in r.text
    assert re.search(r"\d+ notes? links? here", r.text) is not None


def test_atlas_backlink_count_pluralises_the_noun_and_verb_oppositely(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, note_with_links_both_directions
) -> None:
    """Bug hunt 2026-09-18 finding 8: the noun ("note"/"notes") and the
    verb ("links"/"link") need OPPOSITE conditions -- a singular subject
    takes the verb's own "-s" ("1 note links here"), a plural subject
    doesn't ("N notes link here"). The old markup pluralised both off the
    SAME condition, reading as "2 notes links here" (and "1 note link
    here"). Whatever the real backlink count is, the two formulas can
    never agree, so this proves the fixed one -- not just "some digits
    and the words note/link/here" (the existing, looser regex test)."""
    from portolan.web.queries import backlinks

    slug, *_ = note_with_links_both_directions
    count = len(backlinks(demo_conn, slug))
    assert count >= 1

    r = client.get(f"/atlas/{slug}", cookies=auth_cookies)
    assert r.status_code == 200
    correct = f"{count} note{'s' if count != 1 else ''} link{'s' if count == 1 else ''} here"
    buggy = f"{count} note{'s' if count != 1 else ''} link{'' if count == 1 else 's'} here"
    assert correct in r.text
    assert buggy not in r.text


def test_atlas_notes_and_reports_pages_keep_working(
    client: TestClient, auth_cookies, any_claim
) -> None:
    """`/notes/<slug>` isn't replaced by Atlas -- build instructions:
    "Existing /notes/<slug> pages keep working." Reuses the same
    `note_view.build_note_context` data builder Atlas's own note pane
    calls, not a second query path."""
    note_slug, claim_id, *_ = any_claim
    r = client.get(f"/notes/{note_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert f'id="claim-{claim_id}"' in r.text


def test_atlas_tree_groups_by_type_with_signal_account_subgroups(
    demo_conn: psycopg.Connection,
) -> None:
    """The grouping helper behind the tree: one group per note type
    present, ordered per the closed type list, `signal` sub-grouped by
    the account each signal links to (build instructions: "Signals
    sub-grouped by account ... your call")."""
    from portolan.web.atlas_data import atlas_tree_groups

    groups = atlas_tree_groups(demo_conn)
    by_type = {g["type"]: g for g in groups}
    assert "account" in by_type
    assert by_type["account"]["total"] == len(by_type["account"]["notes"])
    assert by_type["account"]["subgroups"] is None

    assert "signal" in by_type
    signal_group = by_type["signal"]
    assert signal_group["notes"] is None
    assert signal_group["subgroups"], "signals should be sub-grouped by account"
    total_signal_notes = sum(len(sub["notes"]) for sub in signal_group["subgroups"])
    assert total_signal_notes == signal_group["total"]
    # Every real account name a signal links to shows up as its own
    # sub-folder label, not the raw slug.
    labels = {sub["label"] for sub in signal_group["subgroups"]}
    assert any(label != "Unlinked" and "-" not in label for label in labels)


def test_atlas_tree_groups_caps_a_folder_at_200_and_flags_truncated(
    demo_conn: psycopg.Connection,
) -> None:
    from portolan.web.atlas_data import atlas_tree_groups

    groups = atlas_tree_groups(demo_conn, cap=1)
    for group in groups:
        if group["total"] > 1:
            shown = (
                len(group["notes"])
                if group["notes"] is not None
                else sum(len(sub["notes"]) for sub in group["subgroups"])
            )
            assert shown <= 1
            assert group["truncated"] is True


def test_atlas_matching_slugs_filters_by_title_and_body(demo_conn: psycopg.Connection) -> None:
    from portolan.web.atlas_data import atlas_matching_slugs

    row = _one(demo_conn, "SELECT slug, title FROM notes WHERE type = 'account' LIMIT 1")
    assert row is not None
    slug, title = row
    needle = title.split()[0]

    matched = atlas_matching_slugs(demo_conn, needle)
    assert slug in matched

    matched_nonsense = atlas_matching_slugs(demo_conn, "zzz-nothing-matches-this-zzz")
    assert matched_nonsense == set()


def test_atlas_matching_slugs_escapes_ilike_wildcards(demo_conn: psycopg.Connection) -> None:
    """Bug hunt 2026-09-18 finding 6: `q` went straight into `%q%` with no
    escaping -- a literal `%` (or `_`) in the search box matched every
    note in the graph instead of a literal percent sign."""
    from portolan.web.atlas_data import atlas_matching_slugs

    total = _one(demo_conn, "SELECT count(*) FROM notes")[0]
    # Unescaped, "%" is a wildcard that matches every row (the finding's
    # own repro: "GET /atlas/panel?q=%25 returns the whole graph").
    assert len(atlas_matching_slugs(demo_conn, "%")) < total
    # A literal "e%" must search for a literal "e%" substring, not "any
    # note containing an e" (which unescaped `%` would match almost every
    # note) -- confirmed indirectly: it must not match strictly more than
    # a plain "e" query would.
    assert atlas_matching_slugs(demo_conn, "e%") <= atlas_matching_slugs(demo_conn, "e")
    from portolan.web.queries import escape_ilike

    assert escape_ilike("100%") == "100\\%"
    assert escape_ilike("a_b") == "a\\_b"
    assert escape_ilike("a\\b") == "a\\\\b"


def test_atlas_tree_groups_true_totals_overrides_the_capped_slug_set_count(
    demo_conn: psycopg.Connection,
) -> None:
    """Bug hunt 2026-09-18 finding 2: the Atlas tree's keyword filter caps
    at `atlas_matching_slugs`' own `limit` (500) matches; a folder's own
    header must show the TRUE total, never "however many of the capped
    set happened to be this type". Proven directly against `atlas_tree_
    groups`' own `true_totals` parameter (the mechanism `routes.atlas`
    wires up from `atlas_matching_type_counts`) rather than seeding
    50,000 notes to trigger the real cap."""
    from portolan.web.atlas_data import atlas_tree_groups

    row = _one(demo_conn, "SELECT slug, type FROM notes WHERE type = 'account' LIMIT 1")
    assert row is not None
    slug, note_type = row

    # A deliberately-wrong true_totals, far larger than the one-slug
    # capped set below -- proves `total` comes from true_totals, not from
    # counting `only_slugs` itself.
    groups = atlas_tree_groups(demo_conn, only_slugs={slug}, true_totals={note_type: 12_500})
    account_group = next(g for g in groups if g["type"] == note_type)
    assert account_group["total"] == 12_500
    assert account_group["truncated"] is True

    # A type present in true_totals but with ZERO rows in the (globally
    # capped) only_slugs set still gets its own folder -- empty, but
    # never silently dropped from the tree. ("decision", not "signal":
    # signal folders are always subgrouped, with `notes` left `None` by
    # design regardless of count -- a different code path than this one.)
    groups2 = atlas_tree_groups(
        demo_conn, only_slugs={slug}, true_totals={note_type: 12_500, "decision": 40}
    )
    decision_group = next(g for g in groups2 if g["type"] == "decision")
    assert decision_group["total"] == 40
    assert decision_group["notes"] == []
    assert decision_group["truncated"] is True


def test_atlas_note_properties_covers_every_frontmatter_field(
    demo_conn: psycopg.Connection,
) -> None:
    """Every frontmatter field a note actually carries becomes its own
    label/value row -- type and status as badge chips, dates as dates,
    source as links (build instructions)."""
    from portolan.notes.persist import read_note
    from portolan.web.atlas_data import note_properties

    row = _one(demo_conn, "SELECT slug FROM notes WHERE type = 'account' LIMIT 1")
    assert row is not None
    (slug,) = row
    note, _claims = read_note(demo_conn, slug)

    rows = note_properties(demo_conn, note, account=None, is_stale=False)
    labels = {r["label"] for r in rows}
    assert {"Type", "Status", "Created", "Updated", "Version"} <= labels
    type_row = next(r for r in rows if r["label"] == "Type")
    assert type_row["kind"] == "badge"
    assert type_row["value"] == "account"
    if note.crm_id:
        assert "CRM ID" in labels


def test_atlas_properties_description_linkifies_wikilinks(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Bug hunt 2026-09-18 finding 3: the Properties block's Description
    row printed a raw `[[slug]]` wikilink instead of a real link into the
    graph -- `atlas_data.py` emits it as `kind: wrap`, and `atlas.html`
    now runs every `wrap` value through the same `linkify_wikilinks`
    filter claim text already uses, instead of `{{ row.value }}` alone."""
    from datetime import UTC, datetime

    from portolan.notes.models import NoteRow, NoteStatus, NoteType
    from portolan.notes.persist import write_note

    target_slug, *_ = any_claim
    now = datetime.now(UTC)
    note = NoteRow(
        slug="bh-description-wikilink-fixture",
        type=NoteType.THEME,
        title="Description wikilink fixture",
        description=f"A facility account, part of [[{target_slug}]].",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Fixture for finding 3.",
    )
    write_note(demo_conn, note, [], changed_by="test", reason="finding-3 fixture")

    r = client.get("/atlas/bh-description-wikilink-fixture", cookies=auth_cookies)
    assert r.status_code == 200
    assert f"[[{target_slug}]]" not in r.text
    assert f'<a href="/notes/{target_slug}">{target_slug}</a>' in r.text


def test_atlas_properties_description_wikilink_to_a_missing_note_renders_as_plain_text(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Demo-polish fix (app checklist 2026-09-18, P0 item 5): unlike
    `parent_slug`/`superseded_by` (both FK-backed columns, bug hunt
    finding 9), a note's `description` is free text with no foreign key
    -- it can genuinely name a `[[slug]]` that was never created or has
    since been deleted. `linkify_wikilinks` now takes the set of slugs
    THIS description actually references that really exist
    (`atlas_data.note_properties`'s own `wiki_targets`); a dangling
    wikilink renders as plain text instead of a link to a 404, matching
    the same "never link to a 404" discipline the Properties block
    already applies elsewhere on this page."""
    from datetime import UTC, datetime

    from portolan.notes.models import NoteRow, NoteStatus, NoteType
    from portolan.notes.persist import write_note

    now = datetime.now(UTC)
    note = NoteRow(
        slug="bh-description-dangling-wikilink-fixture",
        type=NoteType.THEME,
        title="Description dangling wikilink fixture",
        description="A facility account, part of [[this-note-was-never-created]].",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Fixture for the demo-polish wave.",
    )
    write_note(demo_conn, note, [], changed_by="test", reason="polish-item-5 fixture")

    r = client.get("/atlas/bh-description-dangling-wikilink-fixture", cookies=auth_cookies)
    assert r.status_code == 200
    assert "[[this-note-was-never-created]]" not in r.text
    assert '<a href="/notes/this-note-was-never-created">' not in r.text
    assert "this-note-was-never-created" in r.text


def test_atlas_note_edges_grouped_by_kind(
    demo_conn: psycopg.Connection, note_with_links_both_directions
) -> None:
    from portolan.web.atlas_data import note_edges

    slug, outgoing_slug, _incoming_slug = note_with_links_both_directions
    groups = note_edges(demo_conn, slug)
    assert groups, "expected at least one outgoing edge group"
    all_slugs = {entry["slug"] for group in groups for entry in group["entries"]}
    assert outgoing_slug in all_slugs
    # Every group's own kind label is one of the four the build
    # instructions name; `entries`, never `items` (a dict's own `.items`
    # method would otherwise shadow a same-named key under Jinja).
    for group in groups:
        assert group["kind"] in {"body", "parent", "supersession", "provenance"}
        assert "entries" in group


def test_atlas_note_edges_to_a_missing_target_render_as_plain_text_not_a_link(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Bug hunt 2026-09-18 finding 9: `links.to_slug` carries no foreign
    key (`persist._extract_link_targets` writes a row for every
    `[[slug]]` whether or not the target exists) -- an edge to a slug
    that was never written (or has since been deleted) must render as
    plain text, not a link into a 404. Reproduced with a hand-inserted
    `links` row, the same shape a stale `[[slug]]` in a note body leaves
    behind."""
    from portolan.web.atlas_data import note_edges

    note_slug, *_ = any_claim
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, 'body') "
            "ON CONFLICT DO NOTHING",
            (note_slug, "bh-missing-target"),
        )

    groups = note_edges(demo_conn, note_slug)
    body_group = next(g for g in groups if g["kind"] == "body")
    missing = next(e for e in body_group["entries"] if e["slug"] == "bh-missing-target")
    assert missing["href"] is None
    assert missing["title"] == "bh-missing-target"

    r = client.get(f"/atlas/{note_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert 'href="/atlas/bh-missing-target"' not in r.text
    assert "bh-missing-target" in r.text  # still shown, just not linked


def test_atlas_note_properties_source_citation_to_a_missing_source_is_plain_text(
    demo_conn: psycopg.Connection, any_claim
) -> None:
    """Same finding 9, the Properties block's own "Source" chips: a
    `provenance` edge (`links.kind = 'provenance'`) also carries no
    foreign key on `to_slug`, so a citation naming a source slug that was
    never ingested (or has since been removed) must not link to
    `/originals/<slug>`."""
    from portolan.notes.persist import read_note
    from portolan.web.atlas_data import note_properties

    note_slug, *_ = any_claim
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, 'provenance') "
            "ON CONFLICT DO NOTHING",
            (note_slug, "bh-missing-source"),
        )

    note, _claims = read_note(demo_conn, note_slug)
    rows = note_properties(demo_conn, note, account=None, is_stale=False)
    source_row = next(r for r in rows if r["label"] == "Source")
    missing = next(s for s in source_row["value"] if s["slug"] == "bh-missing-source")
    assert missing["exists"] is False


def test_atlas_note_properties_parent_account_shows_the_real_title_not_the_slug(
    demo_conn: psycopg.Connection,
) -> None:
    """Bug hunt 2026-09-18 finding 10: `Parent account`/`Superseded by`
    used to print the bare slug as the link text -- `note_properties` now
    resolves both through `note_summaries`, the same way `Account` always
    has. (`notes.parent_slug` carries a real foreign key -- `ON DELETE
    RESTRICT` -- so unlike the `links` table above, this pointer can
    never reference a note that doesn't exist; the fix here is the title
    resolution, not an existence check.)"""
    from datetime import UTC, datetime

    from portolan.notes.models import NoteRow, NoteStatus, NoteType
    from portolan.notes.persist import write_note
    from portolan.web.atlas_data import note_properties

    now = datetime.now(UTC)
    parent = NoteRow(
        slug="bh-parent-account",
        type=NoteType.ACCOUNT,
        title="Bh Parent Account Title",
        crm_id="BH-PARENT-1",
        description="fixture",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Fixture parent account.",
    )
    write_note(demo_conn, parent, [], changed_by="test", reason="finding-10 fixture")
    child = NoteRow(
        slug="bh-child-account",
        type=NoteType.ACCOUNT,
        title="Bh Child Account Title",
        crm_id="BH-CHILD-1",
        parent_slug="bh-parent-account",
        description="fixture",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Fixture child account.",
    )
    write_note(demo_conn, child, [], changed_by="test", reason="finding-10 fixture")

    from portolan.notes.persist import read_note

    note, _claims = read_note(demo_conn, "bh-child-account")
    rows = note_properties(demo_conn, note, account=None, is_stale=False)
    parent_row = next(r for r in rows if r["label"] == "Parent account")
    assert parent_row["value"] == "Bh Parent Account Title"
    assert parent_row["href"] == "/atlas/bh-parent-account"


def test_atlas_panel_endpoint_returns_a_bare_fragment(client: TestClient, auth_cookies) -> None:
    """`GET /atlas/panel`: the left pane's own fragment, what the tree
    search box's `hx-get` swaps in as the reader types (build
    instructions). Never a full page -- no shell, no `<html>`."""
    r = client.get("/atlas/panel", cookies=auth_cookies)
    assert r.status_code == 200
    assert "<html" not in r.text.lower()
    assert "atlas-folder" in r.text


def test_atlas_panel_search_filter_shows_matching_notes_and_best_matches(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """ "as you type ... the tree filters to matching notes using keyword
    search over title and body ... a 'Best matches' group on top shows
    the existing fused search results with snippets" (build
    instructions)."""
    row = _one(demo_conn, "SELECT slug, title FROM notes WHERE type = 'account' LIMIT 1")
    assert row is not None
    slug, title = row
    needle = title.split()[0]

    r = client.get("/atlas/panel", params={"q": needle}, cookies=auth_cookies)
    assert r.status_code == 200
    assert "Best matches" in r.text
    assert f'href="/atlas/{slug}?q={needle}"' in r.text

    # Bug hunt 2026-09-18 finding 5: a truncated folder's own "Show all N"
    # must carry the active filter through -- otherwise it lands on the
    # UNfiltered `/notes?type=X` (a different, larger N than the one
    # printed). Rendered directly (the demo corpus is too small to trip
    # the real 200-per-folder cap through the live endpoint).
    from portolan.web.templating import build_templates

    template = build_templates().env.get_template("_atlas_panel.html")
    html = template.render(
        q="zebra",
        groups=[
            {
                "type": "action",
                "label": "Actions",
                "total": 622,
                "notes": [{"slug": "a1", "title": "A1", "updated": None}],
                "truncated": True,
                "subgroups": None,
            }
        ],
        best_matches=[],
        active_slug="",
        active_type=None,
        active_subgroup_key=None,
    )
    # Jinja HTML-autoescapes the `&` in the querystring (`&amp;`) -- the
    # same way any other multi-param link in this app's templates does;
    # a browser decodes it back to a real `&` when following the link.
    assert 'href="/notes?type=action&amp;q=zebra"' in html
    assert "Show all 622" in html


def _fake_search_hit(note_slug: str, title: str) -> object:
    from portolan.search.query import SearchHit

    return SearchHit(
        note_slug=note_slug,
        title=title,
        score=1.0,
        via=["fts"],
        best_chunk="a snippet",
        heading=None,
        updated=date.today(),
        source_slugs=[],
    )


def test_atlas_panel_shows_best_matches_without_no_notes_match_when_tree_is_empty(
    client: TestClient, auth_cookies, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug hunt 2026-09-18 finding 7: a query with no exact ILIKE
    substring hit in the tree can still carry fused "Best matches" (the
    vector half of the search can find something the plain-substring tree
    filter doesn't) -- "No notes match" must NOT show while matches are
    listed directly above it. `run_search` is monkeypatched to a
    deterministic non-empty result (rather than relying on the fake
    embedder's incidental behaviour for a nonsense query, which is not a
    reliable invariant across the whole test session's shared corpus)."""
    import portolan.web.routes.atlas as atlas_routes
    from portolan.search.query import SearchResult

    monkeypatch.setattr(
        atlas_routes,
        "run_search",
        lambda *a, **k: SearchResult(
            hits=[_fake_search_hit("some-slug", "Some Title")],
            control_ok=True,
            query_terms_unmatched=[],
        ),
    )

    r = client.get(
        "/atlas/panel", params={"q": "zzz-nothing-matches-this-zzz"}, cookies=auth_cookies
    )
    assert r.status_code == 200
    assert "atlas-folder" not in r.text  # the ILIKE tree really is empty
    assert "Best matches" in r.text  # ...but fused search still found something
    assert "No notes match" not in r.text


def test_atlas_panel_shows_no_notes_match_only_when_nothing_at_all_matches(
    client: TestClient, auth_cookies, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 7's other half: when BOTH the ILIKE tree and the fused
    "Best matches" are genuinely empty, the empty-state line still shows."""
    import portolan.web.routes.atlas as atlas_routes
    from portolan.search.query import SearchResult

    monkeypatch.setattr(
        atlas_routes,
        "run_search",
        lambda *a, **k: SearchResult(hits=[], control_ok=True, query_terms_unmatched=[]),
    )

    r = client.get(
        "/atlas/panel", params={"q": "zzz-nothing-matches-this-zzz"}, cookies=auth_cookies
    )
    assert r.status_code == 200
    assert "atlas-folder" not in r.text
    assert "Best matches" not in r.text
    assert "No notes match" in r.text


def test_atlas_panel_active_slug_marks_current_note(
    client: TestClient, auth_cookies, any_claim
) -> None:
    """The current note is highlighted in the tree, and its folder is
    already expanded (build instructions: "the current note highlighted
    ... the folder of the open note expanded on load")."""
    note_slug, *_ = any_claim
    r = client.get(f"/atlas/{note_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert 'atlas-note-link active"' in r.text
    assert f'href="/atlas/{note_slug}"' in r.text
    assert '<details class="atlas-folder" open>' in r.text


# --------------------------------------------------------------------
# Punch item 4: the weekly report renders markdown, wikilinks resolved.
# --------------------------------------------------------------------


def _write_test_report(
    demo_conn: psycopg.Connection,
    *,
    slug: str,
    week: str,
    target_slug: str,
    extra_body: str = "",
) -> None:
    from datetime import UTC, datetime

    from portolan.notes.models import NoteRow, NoteStatus, NoteType
    from portolan.notes.persist import write_note

    now = datetime.now(UTC)
    body = (
        "### Headline\n"
        f"- Renewal risk rising at [[{target_slug}]]\n\n"
        "### Coverage\n"
        "- Seen 10, extracted 9, partial 0, failed 1, sampled 2\n"
    )
    if extra_body:
        body += f"\n{extra_body}\n"
    note = NoteRow(
        slug=slug,
        type=NoteType.REPORT,
        title=f"Weekly report — {week}",
        description=f"Weekly Portolan report for {week}.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis=body,
        related=[],
    )
    write_note(
        demo_conn, note, [], changed_by="test", reason="test report", embedder=FakeEmbedder()
    )
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO reports (week, note_slug) VALUES (%s, %s) "
            "ON CONFLICT (week) DO UPDATE SET note_slug = EXCLUDED.note_slug",
            (week, slug),
        )
    demo_conn.commit()


def test_report_detail_renders_markdown_and_resolves_wikilinks(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    target_slug, *_ = any_claim
    _write_test_report(demo_conn, slug="report-2099-W01", week="2099-W01", target_slug=target_slug)

    r = client.get("/reports/report-2099-W01", cookies=auth_cookies)
    assert r.status_code == 200
    assert "[[" not in r.text
    assert "### Headline" not in r.text
    assert "<h3>Headline</h3>" in r.text
    assert f'href="/notes/{target_slug}"' in r.text
    # Related / Backlinks / Version history are never shown for a report.
    assert ">Related<" not in r.text
    assert ">Backlinks<" not in r.text
    assert ">Version history<" not in r.text


def test_reports_detail_still_404s_for_a_non_report_note(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    row = _one(demo_conn, "SELECT slug FROM notes WHERE type != 'report' LIMIT 1")
    assert row is not None
    (slug,) = row
    r = client.get(f"/reports/{slug}", cookies=auth_cookies)
    assert r.status_code == 404


# --------------------------------------------------------------------
# Owner review 2026-09-18 (live PDF export, `report-2026-W36`): citation
# markers become real links (never a raw `^[...]` or an exposed url), a
# theme/action name repeated right next to its own just-resolved wikilink
# is stripped even in an ALREADY-STORED report (render-time, no
# regeneration required), and both `.md` downloads exist.
# --------------------------------------------------------------------


def test_report_detail_renders_citation_markers_as_links_with_the_date(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """ "several markdown url links that are raw and exposed instead of
    being hyperlinks." A `^[source @ locator]` marker must render as a
    real link to `/originals/{source}?at={locator}`, labelled with the
    source's own date -- never the raw marker, never the bare locator/url
    (an RSS locator, `portolan.ingest.prepare`'s `url:<link> <date>`
    shape, is exactly the exposed-url case the owner saw)."""
    target_slug, _claim_id, source_slug, locator = any_claim
    row = _one(demo_conn, "SELECT source_date FROM sources WHERE slug = %s", (source_slug,))
    assert row is not None
    (source_date,) = row

    extra = f"### Outliers\n- [[{target_slug}]]: something happened ^[{source_slug} @ {locator}]\n"
    _write_test_report(
        demo_conn,
        slug="report-2099-W02",
        week="2099-W02",
        target_slug=target_slug,
        extra_body=extra,
    )

    r = client.get("/reports/report-2099-W02", cookies=auth_cookies)
    assert r.status_code == 200
    assert "^[" not in r.text
    assert f"@ {locator}" not in r.text
    assert f'href="/originals/{source_slug}?at={quote(locator, safe="")}"' in r.text
    assert f">{source_date:%Y-%m-%d}<" in r.text


def test_note_detail_renders_citation_markers_as_links_with_the_date(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    from datetime import UTC, datetime

    from portolan.notes.models import NoteRow, NoteStatus, NoteType
    from portolan.notes.persist import write_note

    _target_slug, _claim_id, source_slug, locator = any_claim
    row = _one(demo_conn, "SELECT source_date FROM sources WHERE slug = %s", (source_slug,))
    assert row is not None
    (source_date,) = row

    now = datetime.now(UTC)
    slug = "theme-citation-link-fixture"
    note = NoteRow(
        slug=slug,
        type=NoteType.THEME,
        title="Citation link fixture theme",
        description="fixture",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis=f"A customer complained about this. ^[{source_slug} @ {locator}]",
    )
    write_note(
        demo_conn,
        note,
        [],
        changed_by="test",
        reason="citation link fixture",
        embedder=FakeEmbedder(),
    )

    r = client.get(f"/notes/{slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert "^[" not in r.text
    assert f'href="/originals/{source_slug}?at={quote(locator, safe="")}"' in r.text
    assert f">{source_date:%Y-%m-%d}<" in r.text


def test_report_detail_strips_a_stored_repeated_theme_name(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Render-time fix (task: "so existing stored reports improve without
    regeneration") for a report note ALREADY stored in the old
    `[[slug]] "name"` shape -- the owner's own literal live-PDF example,
    "the name of the company is repeated 2x in each line." Not the FIX
    wave's own new-report format (covered in `tests/test_report_weekly.
    py`) -- this proves the render-time cleanup on old stored text."""
    target_slug, *_ = any_claim
    row = _one(demo_conn, "SELECT title FROM notes WHERE slug = %s", (target_slug,))
    assert row is not None
    (target_title,) = row

    extra = (
        "### Themes by movement\n"
        f'- **rising** — [[{target_slug}]] "{target_title}": 7 signals across 1 accounts, '
        "$100 ARR\n"
    )
    _write_test_report(
        demo_conn,
        slug="report-2099-W03",
        week="2099-W03",
        target_slug=target_slug,
        extra_body=extra,
    )

    r = client.get("/reports/report-2099-W03", cookies=auth_cookies)
    assert r.status_code == 200
    assert f'{target_title}"' not in r.text
    # The default headline (`_write_test_report`) links the same slug
    # once more, elsewhere in the body -- 2 total, not 3.
    assert r.text.count(target_title) == 2


def test_report_detail_strips_a_stored_repeated_action_sentence(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Same render-time cleanup, for an already-stored Actions bullet in
    the old `[[slug]]: sentence (owner: ...)` shape."""
    target_slug, *_ = any_claim
    row = _one(demo_conn, "SELECT title FROM notes WHERE slug = %s", (target_slug,))
    assert row is not None
    (target_title,) = row

    extra = f"### Actions\n  - [[{target_slug}]]: {target_title} (owner: unassigned)\n"
    _write_test_report(
        demo_conn,
        slug="report-2099-W04",
        week="2099-W04",
        target_slug=target_slug,
        extra_body=extra,
    )

    r = client.get("/reports/report-2099-W04", cookies=auth_cookies)
    assert r.status_code == 200
    assert r.text.count(target_title) == 2  # the headline's own link, plus this one -- not 3


def test_report_download_returns_attachment_with_absolute_links(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Task step 4: `GET /reports/{slug}.md` -- front matter, absolute
    `[[slug]]`/`^[...]` links, `Content-Disposition: attachment`."""
    target_slug, _claim_id, source_slug, locator = any_claim
    extra = f"### Outliers\n- [[{target_slug}]]: something happened ^[{source_slug} @ {locator}]\n"
    _write_test_report(
        demo_conn,
        slug="report-2099-W05",
        week="2099-W05",
        target_slug=target_slug,
        extra_body=extra,
    )

    r = client.get("/reports/report-2099-W05.md", cookies=auth_cookies)
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="report-2099-W05.md"'
    assert r.text.startswith("---\n")
    assert "generated-at:" in r.text
    assert "[[" not in r.text
    assert "^[" not in r.text
    assert f"http://testserver/notes/{target_slug}" in r.text
    assert f"http://testserver/originals/{source_slug}" in r.text


def test_report_download_keeps_signals_as_a_plain_nested_bullet_list(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Weekly-report wave 2: the `.md` download's own treatment of a
    `<!--signals:...-->` block (task: "the same signals appear as a
    nested bullet list under the line") -- the two sentinel comment
    lines are gone, the bullet underneath survives untouched (still
    absolute-linked, same as every other `[[slug]]`/`^[...]` in the
    file), so the raw markdown a person downloads never shows an HTML
    comment at all."""
    target_slug, _claim_id, source_slug, locator = any_claim
    extra = (
        "### Themes by movement\n"
        f"- **rising** — [[{target_slug}]]: 1 signals across 1 accounts, $100 ARR\n"
        "<!--signals:1 signals-->\n"
        f"  - **complaint** — [[{target_slug}]]: something broke "
        f"^[{source_slug} @ {locator}]\n"
        "<!--/signals-->\n"
    )
    _write_test_report(
        demo_conn,
        slug="report-2099-w07-signals",
        week="2099-W07",
        target_slug=target_slug,
        extra_body=extra,
    )

    r = client.get("/reports/report-2099-w07-signals.md", cookies=auth_cookies)
    assert r.status_code == 200
    assert "<!--signals:" not in r.text
    assert "<!--/signals-->" not in r.text
    assert "- **complaint** —" in r.text
    assert f"http://testserver/originals/{source_slug}" in r.text


def test_report_detail_renders_the_signals_fold_as_a_details_element(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """The same block, through the HTML page: a real `<details
    class="signals-fold">`, not the raw `<!--signals:...-->` comment
    text."""
    target_slug, _claim_id, source_slug, locator = any_claim
    extra = (
        "### Themes by movement\n"
        f"- **rising** — [[{target_slug}]]: 1 signals across 1 accounts, $100 ARR\n"
        "<!--signals:1 signals-->\n"
        f"  - **complaint** — [[{target_slug}]]: something broke "
        f"^[{source_slug} @ {locator}]\n"
        "<!--/signals-->\n"
    )
    _write_test_report(
        demo_conn,
        slug="report-2099-w08-signals",
        week="2099-W08",
        target_slug=target_slug,
        extra_body=extra,
    )

    r = client.get("/reports/report-2099-w08-signals", cookies=auth_cookies)
    assert r.status_code == 200
    assert '<details class="signals-fold">' in r.text
    assert "1 signals ▾" in r.text
    assert "<!--signals:" not in r.text


def test_report_download_requires_session(client: TestClient) -> None:
    r = client.get("/reports/report-2099-W05.md", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login"


def test_note_download_returns_attachment_with_absolute_links(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    from datetime import UTC, datetime

    from portolan.notes.models import NoteRow, NoteStatus, NoteType
    from portolan.notes.persist import write_note

    _target_slug, _claim_id, source_slug, locator = any_claim
    now = datetime.now(UTC)
    slug = "theme-download-fixture"
    note = NoteRow(
        slug=slug,
        type=NoteType.THEME,
        title="Download fixture theme",
        description="fixture",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis=f"A customer complained. ^[{source_slug} @ {locator}]",
    )
    write_note(
        demo_conn, note, [], changed_by="test", reason="download fixture", embedder=FakeEmbedder()
    )

    r = client.get(f"/notes/{slug}.md", cookies=auth_cookies)
    assert r.status_code == 200
    assert r.headers["content-disposition"] == f'attachment; filename="{slug}.md"'
    assert r.text.startswith("---\n")
    assert 'title: "Download fixture theme"' in r.text
    assert "^[" not in r.text
    assert f"http://testserver/originals/{source_slug}" in r.text


def test_note_download_includes_claims_as_readable_citations_not_raw_markers(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Demo-polish fix (app checklist 2026-09-18, P0 item 5): a note's
    `.md` download used to carry only `note.synthesis` -- for a signal
    note (whose synthesis is a generic one-liner; the real content is
    its `claims`, same as the HTML page's own "Claims" card shows) the
    download was a near-empty file with no trace of the claim, its
    quote, or its citation at all. Now the download's own `## Claims`
    section carries every claim with a readable `[source: slug @
    locator](url)` citation -- never the internal `^[source @ locator]`
    marker syntax the stored `body_md` itself uses."""
    note_slug, _claim_id, source_slug, locator = any_claim

    r = client.get(f"/notes/{note_slug}.md", cookies=auth_cookies)
    assert r.status_code == 200
    assert "## Claims" in r.text
    # Never the internal citation-marker syntax (`^[source @ locator]`,
    # `notes.render.CITATION_RE` -- requires an `@`). Deliberately NOT a
    # blanket "no `^[` anywhere" check: a hand-edited claim legitimately
    # carries a DIFFERENT, spec-mandated marker with no `@` in it at all
    # -- `^[edit:<editor> <date>]` (`portolan.consolidate.edits`, spec
    # sec 4's own literal rendered form for an edit record) -- which is
    # not a citation and is not this fix's concern.
    assert CITATION_RE.search(r.text) is None
    # A real, readable, clickable citation instead -- the source slug and
    # locator spelled out, wrapped in a real markdown link.
    assert f"[source: {source_slug} @ {locator}]" in r.text
    assert f"http://testserver/originals/{source_slug}?at=" in r.text


def test_note_download_requires_session(client: TestClient) -> None:
    r = client.get("/notes/theme-download-fixture.md", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login"


def test_report_detail_strips_a_self_referential_mention_appended_by_link_mentions(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Owner review 2026-09-18, live regeneration against `portolan_
    demo4`: `portolan.notes.render.link_mentions`'s own fallback ("a
    mention not found verbatim ... is appended parenthetically at the
    end") can store a claim's own account a SECOND time, past the
    Outliers/Contact-changes bullet's own `[[acct]]:` prefix -- e.g.
    `- [[larkspur-regional-health]]: Larkspur Regional Health appointed
    ... ([[larkspur-regional-health]]) ^[...]`. The trailing, exactly-
    self-referential parenthetical is stripped at render time; an
    unrelated mention (a different slug) in the same position is not."""
    target_slug, _claim_id, source_slug, locator = any_claim
    row = _one(demo_conn, "SELECT slug FROM notes WHERE type = 'product' LIMIT 1")
    other_slug = row[0] if row else "some-other-note"

    extra = (
        "### Outliers\n"
        f"- [[{target_slug}]]: Something happened at the account. "
        f"([[{target_slug}]]) ^[{source_slug} @ {locator}]\n"
        f"- [[{target_slug}]]: A different mention worth keeping. "
        f"([[{other_slug}]]) ^[{source_slug} @ {locator}]\n"
    )
    _write_test_report(
        demo_conn,
        slug="report-2099-W06",
        week="2099-W06",
        target_slug=target_slug,
        extra_body=extra,
    )

    r = client.get("/reports/report-2099-W06", cookies=auth_cookies)
    assert r.status_code == 200
    assert "Something happened at the account." in r.text

    # The self-referential trailing link is gone: nothing between this
    # bullet's own sentence and its "Open the original" citation link
    # points back at `target_slug` a second time.
    after_first = r.text.split("Something happened at the account.", 1)[1]
    before_citation = after_first.split("/originals/", 1)[0]
    assert f'href="/notes/{target_slug}"' not in before_citation

    # An unrelated mention (a different slug) in the same trailing
    # position is never touched -- it still links through.
    after_second = r.text.split("A different mention worth keeping.", 1)[1]
    before_citation_2 = after_second.split("/originals/", 1)[0]
    assert f'href="/notes/{other_slug}"' in before_citation_2


# --------------------------------------------------------------------
# Review finding 1 (blocker): the report renderer must never pass raw
# HTML through -- proven end to end via both the pipeline path (a
# report's synthesis lifted from claim text) and the edit form (any
# logged-in reader rewriting a report's synthesis via
# `POST /notes/{slug}/edit`).
# --------------------------------------------------------------------

_XSS_PAYLOAD = "<img src=x onerror=\"alert(document.cookie)\"><script>fetch('/graph.json')</script>"


def test_report_body_html_never_passes_raw_script_or_img_onerror() -> None:
    """Unit-level proof against `render_report_body` itself: `markdown-it-
    py`'s `commonmark` preset defaults `html: True` -- passing no options
    dict at all (the bug this finding fixed) reproduces the leak; the
    fixed module must render both tags as inert text."""
    from portolan.web.markdown import render_report_body

    html = str(render_report_body(_XSS_PAYLOAD, {}))
    assert "<script>" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img" in html


# --------------------------------------------------------------------
# Weekly-report wave 2 ("make signals visible everywhere", owner review
# 2026-09-18): a report's "N signals" line marks its full signal list
# with `<!--signals:LABEL-->...<!--/signals-->` (`portolan.report.
# weekly._signals_fold_block`); `render_report_body` must turn that into
# a real `<details class="signals-fold">` fold with a chip-styled,
# hover-previewed `<summary>`, and it must stay exactly as safe against a
# stray `<script>`/`onerror` in a claim's own text as the rest of the
# body already is.
# --------------------------------------------------------------------


def _signals_fold_body(inner_bullet: str, *, label: str = "2 signals") -> str:
    return (
        "### Themes by movement\n"
        "- **rising** — [[theme-slug]]: 2 signals across 1 accounts (a-acct), $100 ARR\n"
        f"<!--signals:{label}-->\n"
        f"{inner_bullet}\n"
        "<!--/signals-->\n"
    )


def test_signals_fold_renders_as_a_details_chip_with_a_hover_preview() -> None:
    from portolan.web.markdown import render_report_body

    body = _signals_fold_body(
        "  - **complaint** — [[a-acct]]: Something broke badly. "
        "^[S-1 @ loc:1] · [[signal-complaint-a-acct]]"
    )
    titles = {"theme-slug": "Theme Title", "a-acct": "A Account"}
    html = str(render_report_body(body, titles, {}))

    assert '<details class="signals-fold">' in html
    assert '<summary class="badge"' in html
    assert "2 signals ▾</summary>" in html
    assert "title=" in html
    # The fold is a nested child of the theme's own bullet, not a block
    # that breaks the surrounding list -- same nesting shape the old
    # "up to 3 representative quotes" preview had.
    assert "<!--signals:" not in html
    assert "<!--/signals-->" not in html


def test_signals_fold_escapes_xss_in_a_signals_claim_text() -> None:
    """The XSS payload can arrive inside a signal's own `claim_text`
    (lifted verbatim from a customer email/ticket, same as any other
    claim in the report) -- it must render exactly as inert as it does
    everywhere else in the body (review finding 1)."""
    from portolan.web.markdown import render_report_body

    body = _signals_fold_body(f"  - **complaint** — [[a-acct]]: {_XSS_PAYLOAD} ^[S-1 @ loc:1]")
    html = str(render_report_body(body, {"a-acct": "A Account"}, {}))

    assert "<script>fetch(" not in html
    assert "<img " not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img" in html


def test_signals_fold_hover_preview_has_no_markdown_or_html_markup() -> None:
    """The `title=` tooltip is plain text a browser shows verbatim in its
    own native popup -- it must never carry `**`/`[[...]]` markdown
    syntax a reader would see literally, and any `<`/`>` in the source
    text must still come through escaped (it's an HTML attribute)."""
    from portolan.web.markdown import render_report_body

    body = _signals_fold_body(
        "  - **complaint** — [[a-acct]]: Reported <b>issue</b> here. ^[S-1 @ loc:1]"
    )
    html = str(render_report_body(body, {"a-acct": "A Account"}, {}))

    title_start = html.index('title="') + len('title="')
    title_end = html.index('"', title_start)
    preview = html[title_start:title_end]
    assert "**" not in preview
    assert "[[" not in preview
    assert "<b>" not in preview  # the raw tag must not survive unescaped
    assert "&lt;b&gt;" in preview


def test_signals_fold_missing_from_a_note_with_no_such_block_is_a_no_op() -> None:
    """A report generated before this wave (or any ordinary note body)
    has no `<!--signals:...-->` marker at all -- must render exactly as
    it always did, no stray `<details>` anywhere."""
    from portolan.web.markdown import render_report_body

    body = "### Headline\n- Renewal risk rising at [[some-account]]\n"
    html = str(render_report_body(body, {"some-account": "Some Account"}, {}))
    assert "<details" not in html
    assert "signals-fold" not in html


def test_unpaired_signals_sentinel_does_not_leak_onto_the_page(client: TestClient) -> None:
    """Bug hunt 2026-09-18 finding 13: a hand edit that deletes one half
    of a `<!--signals:...-->` / `<!--/signals-->` pair (e.g. via
    `/notes/{slug}/edit`) used to leave the surviving sentinel line
    escaped and SHOWN, verbatim, on the page -- reproduced with the exact
    payload from the finding's own repro."""
    from portolan.web.markdown import render_report_body

    body = "- **Theme A** happened.\n<!--signals:3 signals-->\n  - Signal A1: something.\n"
    html = str(render_report_body(body, {}))
    assert "signals" not in html.lower()
    assert "&lt;!--" not in html
    assert "Theme A" in html  # the rest of the body still renders normally
    assert "Signal A1" in html


def test_unpaired_signals_sentinel_stripped_from_the_md_download_too() -> None:
    """The `.md` download's own treatment: no markdown renderer runs over
    it at all, so an un-stripped sentinel would ship as a raw HTML
    comment in the downloaded file."""
    from portolan.web.markdown import _strip_unpaired_signals_sentinels

    body = "- **Theme A** happened.\n<!--signals:3 signals-->\n  - Signal A1: something.\n"
    stripped = _strip_unpaired_signals_sentinels(body)
    assert "<!--signals" not in stripped
    assert "Theme A" in stripped
    assert "Signal A1" in stripped


def test_note_download_escapes_a_malicious_wikilink_title_from_the_linked_note(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Bug hunt 2026-09-18 finding 11: `[{title}](...)` (the wikilink's
    rendered markdown link) built a linked note's TITLE in with no
    escaping. The HTML pages were already safe (markdown-it-py, `html:
    False`, escapes text nodes) but the raw `.md` download is served as a
    literal string with no processing of its own -- reproduced with the
    finding's own exact payload, a note titled with a raw `<img onerror>`
    tag, linked from a second note's synthesis."""
    from datetime import UTC, datetime

    from portolan.notes.models import NoteRow, NoteStatus, NoteType
    from portolan.notes.persist import write_note

    now = datetime.now(UTC)
    evil_title = '<img src=x onerror="alert(1)">Evil Co'
    target = NoteRow(
        slug="bh-xss-note",
        type=NoteType.THEME,
        title=evil_title,
        description="fixture",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Fixture account with a malicious title.",
    )
    write_note(
        demo_conn,
        target,
        [],
        changed_by="test",
        reason="finding-11 fixture",
        embedder=FakeEmbedder(),
    )
    linker = NoteRow(
        slug="bh-linker",
        type=NoteType.THEME,
        title="Linker fixture theme",
        description="fixture",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="See [[bh-xss-note]] link.",
    )
    write_note(
        demo_conn,
        linker,
        [],
        changed_by="test",
        reason="finding-11 fixture",
        embedder=FakeEmbedder(),
    )

    r = client.get("/notes/bh-linker.md", cookies=auth_cookies)
    assert r.status_code == 200
    # The raw tag delimiters are backslash-escaped (CommonMark: removed by
    # any compliant renderer, never themselves shown) -- never a literal,
    # un-escaped `<img ...>` sitting in the downloaded file.
    assert "[<img" not in r.text  # unescaped -- would be the pre-fix shape
    assert r"[\<img" in r.text  # escaped -- the fix
    assert r"\>Evil Co" in r.text

    # The HTML page path was already safe; confirm this fix didn't change
    # that (markdown-it-py's own escaping still applies unchanged).
    r_html = client.get("/notes/bh-linker", cookies=auth_cookies)
    assert r_html.status_code == 200
    assert "<img src=x onerror" not in r_html.text


def test_report_detail_escapes_a_script_tag_delivered_via_claim_text(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Pipeline path: a report's synthesis is built out of claim text
    extracted verbatim from customer emails/tickets/RSS
    (`portolan.report.weekly`), so a stray `<script>` in a claim quote
    must never survive to `/reports/{slug}` as a real tag. `_write_test_
    report` builds its body the same way `weekly.py` does (a synthesis
    string handed straight to `write_note`); the payload is planted
    directly in that body rather than via a raw `UPDATE`, since
    `synthesis` is parsed out of `notes.body_md` at read time, not a
    column of its own (`portolan.notes.models` module docstring)."""
    target_slug, *_ = any_claim
    _write_test_report(
        demo_conn,
        slug="report-2099-w02-xss",
        week="2099-W02",
        target_slug=target_slug,
        extra_body=_XSS_PAYLOAD,
    )

    r = client.get("/reports/report-2099-w02-xss", cookies=auth_cookies)
    assert r.status_code == 200
    assert _XSS_PAYLOAD not in r.text
    assert "<script>fetch(" not in r.text
    assert "<img " not in r.text
    assert "&lt;script&gt;" in r.text
    assert "&lt;img" in r.text


def test_edit_form_injecting_a_script_into_a_reports_synthesis_is_escaped(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    """Editor path: any logged-in reader can `POST /notes/{report-slug}/
    edit` (no `claim_id` -- editing the note's own synthesis), and that
    text is what every other reader's `/reports/{slug}` next renders."""
    target_slug, *_ = any_claim
    _write_test_report(
        demo_conn, slug="report-2099-w03-xss", week="2099-W03", target_slug=target_slug
    )

    r = client.post(
        "/notes/report-2099-w03-xss/edit",
        data={"after_text": f"### Headline\n{_XSS_PAYLOAD}", "reason": "xss test"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r.status_code == 303

    detail = client.get("/reports/report-2099-w03-xss", cookies=auth_cookies)
    assert detail.status_code == 200
    assert _XSS_PAYLOAD not in detail.text
    assert "<script>fetch(" not in detail.text
    assert "<img " not in detail.text
    assert "&lt;script&gt;" in detail.text
    assert "&lt;img" in detail.text


# --------------------------------------------------------------------
# Punch item 5: the coverage table never overflows the page.
# --------------------------------------------------------------------


def test_coverage_page_has_scroll_container_around_every_table(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    with demo_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, extraction_status, s3_key)
            VALUES ('S-web-scroll-test', 'call', 'failed', 'originals/call/S-web-scroll-test')
            ON CONFLICT (slug) DO NOTHING
            """
        )
    r = client.get("/coverage", cookies=auth_cookies)
    assert r.status_code == 200
    assert 'class="table-scroll"' in r.text
    assert r.text.count('class="table-scroll"') >= 2  # ledger + not-ingested, at minimum
    assert 'role="region"' in r.text


# --------------------------------------------------------------------
# Punch item 6: metadata is grouped, not a run-on line.
# --------------------------------------------------------------------


def test_note_page_metadata_is_a_grouped_meta_row(
    client: TestClient, auth_cookies, any_claim
) -> None:
    note_slug, *_ = any_claim
    r = client.get(f"/notes/{note_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert 'class="meta-row"' in r.text
    assert re.search(r'<span class="updated (fresh|stale)">updated: ', r.text)


# --------------------------------------------------------------------
# Punch item 8: the claim line leads with the "dig deeper" chip.
# --------------------------------------------------------------------


def test_claim_line_has_accent_dig_chip_and_quiet_edit(
    client: TestClient, auth_cookies, any_claim
) -> None:
    note_slug, _claim_id, _source_slug, _locator = any_claim
    r = client.get(f"/notes/{note_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert "Open the original at" in r.text
    assert 'class="btn chip dig"' in r.text
    assert 'class="btn quiet"' in r.text
    assert re.search(r'class="badge (fresh|superseded)"', r.text)


# --------------------------------------------------------------------
# Punch item 9: the original viewer never doubles a timestamp.
# --------------------------------------------------------------------


def test_call_transcript_block_never_prints_its_timestamp_twice() -> None:
    from portolan.extract.schema import SourceType
    from portolan.ingest.models import RawSource
    from portolan.web.routes.originals import _blocks_for

    raw = RawSource(
        source_id="S-test-call",
        source_type=SourceType.CALL,
        source_date=date(2026, 1, 1),
        text="[00:01:02] Rob Tallis: Hello there\n[00:01:05] Someone: Hi back\n",
    )
    blocks = _blocks_for(raw)
    assert blocks, "expected at least one block"
    for block in blocks:
        if block["locator"] == "header":
            continue
        assert not block["text"].startswith("["), (
            f"block for locator {block['locator']!r} still carries a leading timestamp: "
            f"{block['text']!r}"
        )
    assert blocks[-1]["text"] == "Someone: Hi back"


def test_original_call_viewer_renders_locator_gutter_once(
    client: TestClient, auth_cookies, call_source
) -> None:
    slug, locator = call_source
    r = client.get(f"/originals/{slug}", params={"at": locator}, cookies=auth_cookies)
    assert r.status_code == 200
    assert 'class="loc"' in r.text
    # The block for this exact locator shouldn't repeat the bracketed
    # timestamp inside its own text span.
    block_match = re.search(
        r'data-locator="' + re.escape(locator) + r'"[^>]*>(.*?)</div>', r.text, re.DOTALL
    )
    assert block_match is not None
    assert block_match.group(1).count("[" + locator + "]") == 0


# --------------------------------------------------------------------
# Punch item 10: loading states for chat and HTMX swaps.
# --------------------------------------------------------------------


def test_chat_page_has_ask_status_indicator(client: TestClient, auth_cookies) -> None:
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    assert 'id="ask-status"' in r.text
    assert 'aria-live="polite"' in r.text


def test_streamed_answer_builds_a_msg_actions_row_with_copy_and_regenerate(
    client: TestClient, auth_cookies
) -> None:
    """Review finding 14: a streamed answer never got a `.msg-actions`
    row at all -- `streamAnswer` only ever built `.msg-text`, so Copy and
    Regenerate appeared only after a full page reload. Built inside the
    `done` handler now, in the same shape a reload's `_chat_sentence.html`
    based render gets."""
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    done_start = r.text.index("parsed.event === 'done'")
    done_body = r.text[done_start : r.text.index("} else if (parsed.event === 'error')")]
    assert "actions.className = 'msg-actions'" in done_body
    assert "'data-action', 'copy-message'" in done_body
    assert "'data-action', 'regenerate'" in done_body
    assert "assistantDiv.appendChild(actions)" in done_body


def test_chat_auto_scroll_uses_scroll_into_view_not_a_no_op(
    client: TestClient, auth_cookies
) -> None:
    """Review finding 15: `.chat-thread` has no `overflow` of its own (the
    page scrolls, not that element), so `thread.scrollTop =
    thread.scrollHeight` was always a no-op."""
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    assert "thread.scrollTop" not in r.text
    assert "scrollIntoView({ block: 'end' })" in r.text


def test_graph_auto_rotate_note_hidden_with_the_button_on_mode_switch(
    client: TestClient, auth_cookies
) -> None:
    """Review finding 16: leaving 3D used to hide the Auto-rotate button
    but leave its reduced-motion note on screen next to it."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    set_mode_start = r.text.index("function setMode(mode) {")
    set_mode_body = r.text[set_mode_start : r.text.index("els.mode2d.addEventListener")]
    assert "els.autoRotateNote.hidden = true" in set_mode_body


# --------------------------------------------------------------------
# Graph v3 (owner review 2026-09-15): click focus stops rotation and
# centres the node, full labels on focus, idle drift, label halo.
# --------------------------------------------------------------------


def test_graph_v3_click_focus_stops_rotation_and_centres_the_node(
    client: TestClient, auth_cookies
) -> None:
    """Item 1: selecting a node stops auto-rotate immediately and eases
    the camera to frame the node plus its direct neighbours, offset for
    the note panel -- in both 2D and 3D."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert "function focusCamera3D(node) {" in text
    assert "function focusCamera2D(node) {" in text
    # zoomToFit(durationMs, padding, nodeFilterFn) framing node + direct
    # neighbours -- used identically by both engines.
    assert text.count("engine.zoomToFit(FOCUS_EASE_MS, 70,") == 2
    # autoRotateStep's own per-frame guard stops rotation the instant a
    # node is selected, regardless of the toggle's own on/off state.
    auto_rotate_start = text.index("function autoRotateStep(t) {")
    auto_rotate_body = text[auto_rotate_start : auto_rotate_start + 400]
    assert "state.selectedId != null" in auto_rotate_body
    # deselectNode only resumes rotation if the toggle itself is still on
    # -- "unless the reader had paused it themselves".
    deselect_start = text.index("function deselectNode() {")
    deselect_body = text[deselect_start : text.index("function renderMeta")]
    assert "startAutoRotateLoop();" in deselect_body


def test_graph_v3_full_labels_on_focus_and_hover(client: TestClient, auth_cookies) -> None:
    """Item 2: a node's own label, and its direct neighbours' labels, show
    the FULL title while selected; a hovered node also shows its full
    label; both truncate again once cleared."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert "function isLabelExpanded(id) {" in text
    assert (
        "return id === state.hoveredId || id === state.selectedId "
        "|| state.selectedNeighborIds.has(id);" in text
    )
    # v4 (owner review 2026-09-15, item 4) replaced the v3 measureText
    # binary-search truncation (`truncateToWidth`) with a plain
    # character-count cut -- see test_graph_v4_smaller_labels below for
    # the full assertion on that replacement.
    assert "function truncateToChars(text, maxChars) {" in text


def test_graph_v3_idle_drift_respects_reduced_motion(client: TestClient, auth_cookies) -> None:
    """Item 3: a small continuous per-node sinusoidal offset, applied on
    top of the simulation position each frame without accumulating, gated
    off entirely under prefers-reduced-motion."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert "function driftStep(t) {" in text
    assert "driftEnabled = !reducedMotion();" in text
    assert "if (driftEnabled) startDriftLoop();" in text
    # never-accumulating: each frame undoes exactly what it added last
    # frame before computing a fresh offset.
    drift_start = text.index("function driftStep(t) {")
    drift_body = text[drift_start : drift_start + 900]
    assert "n.x -= n.__driftOffsetX;" in drift_body


def test_graph_v3_label_halo_stroke_before_fill(client: TestClient, auth_cookies) -> None:
    """Item 4: every label gets a ground-colored stroke drawn behind the
    fill -- the 3D sprite texture and the 2D canvas label both do this,
    with the same round line join."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert text.count("ctx.strokeStyle = graphGround();") == 2
    assert text.count("ctx.lineJoin = 'round';") == 2
    assert text.count("strokeText(") == 2
    # strokeText must precede fillText at each of the two call sites, not
    # just appear somewhere in the file (stroke drawn BEHIND the fill).
    pairs = [
        ("ctx.strokeText(text, tx, ty);", "ctx.fillText(text, tx, ty);"),
        ("ctx.strokeText(text, padding, 32);", "ctx.fillText(text, padding, 32);"),
    ]
    for stroke_call, fill_call in pairs:
        assert text.index(stroke_call) < text.index(fill_call)


# --------------------------------------------------------------------
# Graph v4 (owner review 2026-09-15, second side-by-side pass against
# aicodingdictionary.com): a true-black ramp end, per-node depth fade,
# roughly doubled rest-state spacing, smaller/shorter-cut labels, and a
# cleaner selected state (everyone else fades to gray and loses its
# label).
# --------------------------------------------------------------------


def test_graph_v4_true_black_ramp_ends(client: TestClient, auth_cookies) -> None:
    """Item 1: the grayscale ramp's dark extreme is near-black (`#0e0e0e`)
    in light theme, near-white (`#f2f2f0`) in dark theme -- not the old
    charcoal `#1a1a19` -- and both ramp ends (`monoTokens`) and the
    selected node's forced ink (item 5) cite the same two constants."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert "var TRUE_INK_LIGHT = '#0e0e0e';" in text
    assert "var TRUE_INK_DARK = '#f2f2f0';" in text
    assert "'#1a1a19'" not in text, "the old charcoal ramp end must be gone, not just supplemented"
    mono_start = text.index("function monoTokens() {")
    mono_body = text[mono_start : text.index("function", mono_start + 10)]
    assert "var rampFrom = trueInk();" in mono_body
    # the light end (`#cfcfcd`) and the dark theme's own far end
    # (`--pl-line`) are untouched -- only the dark/near extreme moved.
    assert "'#cfcfcd'" in mono_body


def test_graph_v4_depth_fade_derived_from_the_clouds_own_bounding_radius(
    client: TestClient, auth_cookies
) -> None:
    """Item 2: near nodes render full shade, far nodes fade toward the
    ground -- computed per animation frame (not just per simulation tick,
    which stops firing once the layout settles but the camera keeps
    moving), from a near/far range derived from the live cloud's own
    bounding radius, applied as sprite opacity (never re-rasterizing a
    texture every frame). 2D has no camera depth and is explicitly
    skipped."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert "function cloudBoundingRadius(nodes) {" in text
    assert "function updateDepthFade() {" in text
    fade_start = text.index("function updateDepthFade() {")
    fade_body = text[fade_start : text.index("function depthFadeStep")]
    assert "engineMode !== '3d'" in fade_body, (
        "must be a 3D-only mechanism -- 2D has no camera depth"
    )
    assert "var near = Math.max(1, camDist - radius);" in fade_body
    assert "var far = camDist + radius;" in fade_body
    assert "obj.__disc.material.opacity" in fade_body
    assert "obj.__label.material.opacity" in fade_body
    # per-frame, not per-tick: its own requestAnimationFrame loop, started
    # when 3D constructs/reuses an engine and stopped in disposeEngine.
    assert "function depthFadeStep() {" in text
    assert "depthFadeFrameId = requestAnimationFrame(depthFadeStep);" in text
    assert "stopDepthFadeLoop(); // v4, item 2" in text


def test_graph_v4_spacing_doubled_via_d3_force(client: TestClient, auth_cookies) -> None:
    """Item 3: charge repulsion and link distance are raised directly on
    the underlying d3-force simulation, in BOTH engines -- `.d3Force()`
    is the API both pinned libraries (force-graph and 3d-force-graph)
    expose for this, confirmed directly against the pinned jsdelivr
    bundles (this wave's own research, not assumed)."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert "var LINK_DISTANCE = 60;" in text
    assert "var CHARGE_STRENGTH = -140;" in text
    assert text.count("g.d3Force('link').distance(LINK_DISTANCE);") == 2
    assert text.count("g.d3Force('charge').strength(CHARGE_STRENGTH);") == 2
    # re-tuned rest-state pull-in (both engines) and the static Fog
    # constants (superseded by the dynamic per-node fade) are gone.
    assert "g.zoom(g.zoom() * 1.4, 0);" in text
    assert "x: p.x * 0.68, y: p.y * 0.68, z: p.z * 0.68" in text
    # the old fixed-threshold static fog (superseded by the dynamic
    # per-node fade above) is gone as real code -- only mentioned in this
    # wave's own explanatory comment on why it was removed.
    assert "new THREE.Fog(" not in text


def test_graph_v4_smaller_labels_fixed_scale_and_longer_char_cut(
    client: TestClient, auth_cookies
) -> None:
    """Item 4: every 3D label sprite scales by ONE fixed mid-size-node
    ratio (not each node's own radius, which is what let hub labels grow
    large enough to span and collide with neighbouring nodes), and
    rest-state truncation is a plain ~22-character cut with a plain
    ellipsis instead of a measured-pixel-width cutoff."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert "var LABEL_PX_TO_WORLD = LABEL_MID_NODE_R / 32;" in text
    # used at construction AND at every texture rebuild -- never the
    # node's own per-instance radius again.
    assert text.count("built.width * LABEL_PX_TO_WORLD, built.height * LABEL_PX_TO_WORLD") == 2
    assert "var LABEL_REST_MAX_CHARS = 22;" in text
    assert "function truncateToChars(text, maxChars) {" in text
    assert "return text.slice(0, maxChars) + '...';" in text
    # both engines' rest-state call sites route through the new function.
    assert "truncateToChars(full, LABEL_REST_MAX_CHARS)" in text


def test_graph_v4_selected_state_fades_everyone_else_and_hides_their_labels(
    client: TestClient, auth_cookies
) -> None:
    """Item 5: while a node is selected, its own label is forced to true
    ink at full opacity; its direct neighbours keep their normal shade and
    full labels; every OTHER node fades to a flat gray and loses its
    label entirely -- in both 2D (a live color accessor, re-evaluated
    every frame) and 3D (a texture rebuilt only when the fade state
    changes)."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert "function isFadedBySelection(id) {" in text
    assert (
        "return state.selectedId != null && id !== state.selectedId "
        "&& !state.selectedNeighborIds.has(id);" in text
    )
    assert "function deselectedFadeColor() { return isDarkMode() ? '#2b2b2d' : '#d4d4d2'; }" in text

    # colorFor is the single, shared color accessor -- both engines route
    # a faded node's disc color through the exact same function, so 2D
    # and 3D can never disagree on what "faded" looks like.
    color_for_start = text.index("function colorFor(colors) {")
    color_for_body = text[color_for_start : text.index("}", text.index("};", color_for_start))]
    assert "if (isFadedBySelection(d.id)) return deselectedFadeColor();" in color_for_body

    # 2D: the label draw returns early (no label at all) for a faded
    # node, checked BEFORE isLabelExpanded so hover can't override it.
    node_canvas_start = text.index(".nodeCanvasObject(function (node, ctx, globalScale) {")
    node_canvas_body = text[
        node_canvas_start : text.index(".onEngineTick(function () {", node_canvas_start)
    ]
    assert "if (isFadedBySelection(node.id)) return;" in node_canvas_body
    assert "ctx.fillStyle = isSelected ? trueInk() : c.ink;" in node_canvas_body
    assert "ctx.globalAlpha = isSelected ? 1 : labelOpacity(node.val);" in node_canvas_body

    # 3D: syncSelection3D rebuilds the disc texture on a fade-state
    # change and hides the label sprite outright while faded.
    sync_start = text.index("function syncSelection3D(nodes, links, colors) {")
    sync_body = text[sync_start : text.index("links.forEach(function (l) {", sync_start)]
    assert (
        "obj.__disc.material.map = makeCircleTexture(faded ? deselectedFadeColor() : discColorKey);"
        in sync_body
    )
    assert "obj.__label.visible = !faded2;" in sync_body
    assert "var inkColor = isSelected ? trueInk() : (obj.__labelColor || colors.ink);" in sync_body


def test_regenerate_click_handler_keeps_the_old_bubble_until_the_replacement_succeeds(
    client: TestClient, auth_cookies
) -> None:
    """Review finding 3 (frontend half): `msgDiv.remove()` used to run
    before `streamAnswer` was even called, so a transient model failure
    left the reader looking at nothing. The removal must now happen
    inside `streamAnswer`'s `onDone` callback -- after `streamAnswer(`
    textually, in the same handler -- so the old answer survives an
    error and disappears only once the replacement is confirmed."""
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    handler_start = text.index(
        'data-action="regenerate"]\');', text.index("document.addEventListener")
    )
    handler = text[handler_start : handler_start + 1200]
    stream_call_index = handler.index("streamAnswer('/chat/' + convId2 + '/regenerate'")
    remove_index = handler.index("msgDiv.remove();", stream_call_index)
    assert remove_index > stream_call_index, (
        "the old bubble must be removed from inside streamAnswer's onDone callback, "
        "not before the regenerate request is even sent"
    )


def test_composer_error_handler_never_renders_a_raw_response_body(
    client: TestClient, auth_cookies
) -> None:
    """Bug hunt 2026-09-18, finding 3: a non-2xx `fetch()` response used
    to have its raw body (`{"detail":"..."}` -- the server's actual JSON
    error, escaped but otherwise verbatim) printed straight into the
    answer bubble. `showComposerError` now redirects on 401, shows a
    fixed "no longer exists" sentence on 404, and otherwise pulls
    `.detail` out of a parsed JSON body (or falls back to one generic
    sentence) -- never the raw text."""
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert "function showComposerError" in text
    # The old bug, byte for byte: `escapeHtml(t || ('Error ' + resp.status))`
    # interpolated directly into the bubble's innerHTML.
    assert "escapeHtml(t ||" not in text
    assert "location.href = '/login" in text
    assert "This chat no longer exists" in text
    assert "JSON.parse(rawBody)" in text


def test_composer_error_offers_a_retry_that_resends_the_same_request(
    client: TestClient, auth_cookies
) -> None:
    r = client.get("/", cookies=auth_cookies)
    text = r.text
    assert "retryBtn.addEventListener('click', retry)" in text
    assert "streamAnswer(url, formData, onDone);" in text


def test_copy_button_only_claims_success_after_an_actual_copy(
    client: TestClient, auth_cookies
) -> None:
    """Bug hunt 2026-09-18, finding 11: the Copy button used to set its
    label to "Copied" unconditionally, even when `navigator.clipboard`
    doesn't exist at all (any non-secure origin -- the exact case
    `PORTOLAN_COOKIE_SECURE=false` exists for), so it claimed success
    while copying nothing. The label now only flips inside the async
    clipboard write's own success callback, or after a real
    `execCommand('copy')` fallback -- and says "Copy failed" when
    neither works."""
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    text = r.text
    assert "navigator.clipboard.writeText(text).then(" in text
    assert "execCommand('copy')" in text
    assert "Copy failed" in text


def test_conversation_sidebar_forms_carry_an_htmx_indicator(
    client: TestClient, auth_cookies
) -> None:
    client.post("/conversations", cookies=auth_cookies, follow_redirects=False)
    r = client.get("/", cookies=auth_cookies)
    assert r.status_code == 200
    assert 'class="htmx-indicator' in r.text
    assert "hx-indicator=" in r.text


def test_global_htmx_indicator_css_present(client: TestClient, auth_cookies) -> None:
    r = client.get("/", cookies=auth_cookies)
    assert ".htmx-indicator" in r.text
    assert "htmx-request" in r.text


# --------------------------------------------------------------------
# Punch item 11: branded error pages, never raw JSON, for a browser.
# --------------------------------------------------------------------


def test_404_page_is_branded_html_not_json(client: TestClient, auth_cookies) -> None:
    r = client.get("/notes/does-not-exist-at-all", cookies=auth_cookies)
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "Not found" in r.text
    assert (
        "does-not-exist-at-all" not in r.text or "{" not in r.text.split("does-not-exist")[0][-5:]
    )
    assert '"detail"' not in r.text
    assert 'href="/search"' in r.text
    assert 'href="/"' in r.text


def test_unregistered_path_404s_as_branded_html(client: TestClient, auth_cookies) -> None:
    r = client.get("/this-route-does-not-exist", cookies=auth_cookies)
    assert r.status_code == 404
    assert "Not found" in r.text
    assert '"detail"' not in r.text


def test_500_page_is_branded_html_and_never_leaks_a_traceback(
    demo_conn: psycopg.Connection,
    db_url: str,
    store_root: Path,
    any_claim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dedicated app/client, `raise_server_exceptions=False` (like
    `test_session_cookie_is_secure_httponly_samesite_lax`'s own dedicated
    client, above): Starlette's `ServerErrorMiddleware` re-raises an
    unhandled exception after sending the handler's response, precisely
    so a real ASGI server still logs it -- the shared `client` fixture's
    default `TestClient` would surface that re-raise as a test failure
    even though the branded 500 page was correctly sent on the wire."""
    from portolan.web import note_view as note_view_module

    note_slug, *_ = any_claim

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic failure for the 500-page test, never shown to a reader")

    monkeypatch.setattr(note_view_module, "read_note", _boom)

    settings = _settings(db_url, store_root)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app, raise_server_exceptions=False) as c:
        login = c.post(
            "/login", data={"org_key": ORG_KEY, "editor": "500 Tester"}, follow_redirects=False
        )
        r = c.get(f"/notes/{note_slug}", cookies=login.cookies)

    assert r.status_code == 500
    assert r.headers["content-type"].startswith("text/html")
    assert "Something went wrong" in r.text
    assert "Traceback" not in r.text
    assert "synthetic failure" not in r.text
    assert '"detail"' not in r.text


def test_500_responses_carry_the_csp_and_baseline_security_headers(
    demo_conn: psycopg.Connection,
    db_url: str,
    store_root: Path,
    any_claim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starlette serves an unhandled exception from `ServerErrorMiddleware`,
    outside the CSP middleware, so the 500 handler adds the headers itself.
    The page's inline scripts must carry the header's nonce, or the browser
    refuses them."""
    from portolan.web import note_view as note_view_module

    note_slug, *_ = any_claim

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic failure for the 500 security-header test")

    monkeypatch.setattr(note_view_module, "read_note", _boom)

    settings = _settings(db_url, store_root)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: demo_conn
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app, raise_server_exceptions=False) as c:
        login = c.post(
            "/login", data={"org_key": ORG_KEY, "editor": "500 Tester"}, follow_redirects=False
        )
        page = c.get(f"/notes/{note_slug}", cookies=login.cookies)
        fragment = c.get(
            f"/notes/{note_slug}", cookies=login.cookies, headers={"HX-Request": "true"}
        )

    assert page.status_code == 500
    csp = page.headers.get("content-security-policy", "")
    nonce_match = re.search(r"'nonce-([^']+)'", csp)
    assert nonce_match is not None, csp
    assert f'nonce="{nonce_match.group(1)}"' in page.text
    assert "frame-ancestors 'self'" in csp

    assert fragment.status_code == 500
    assert "content-security-policy" in fragment.headers
    for response in (page, fragment):
        assert response.headers.get_list("x-content-type-options") == ["nosniff"]
        assert response.headers.get("cache-control") == "private, no-store"
        assert response.headers.get("referrer-policy") == "same-origin"


def test_chat_stream_error_stays_plain_text_not_a_branded_page(
    client: TestClient, auth_cookies
) -> None:
    """The one deliberate carve-out (`app.py`'s `_wants_json_error`): the
    chat composer's hand-rolled `fetch()` reads this body directly as the
    error text to show inline -- it must never become a full HTML page."""
    r = client.post("/chat/new/stream", data={"question": "x" * 32_001}, cookies=auth_cookies)
    assert r.status_code == 400
    assert "<html" not in r.text.lower()


def test_htmx_fragment_error_stays_json_not_a_branded_page(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    with demo_conn.cursor() as cur:
        cur.execute("INSERT INTO conversations (owner) VALUES ('Not You') RETURNING id")
        (other_id,) = cur.fetchone()
    demo_conn.commit()
    r = client.post(
        f"/conversations/{other_id}/rename",
        data={"title": "Hijacked"},
        cookies=auth_cookies,
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 404
    assert "<html" not in r.text.lower()


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 2: a 401 on a plain browser navigation
# (no `HX-Request`, `Accept` prefers `text/html` -- the signal a real
# `<form method=post>` submit sends, matching `chat.html`'s own
# "+ New chat" button) now redirects to `/login?next=<path>` instead of
# dead-ending on raw JSON. Every API/htmx/fetch caller (none of which
# send that `Accept` header) is unaffected -- proven directly below, and
# by every pre-existing 401-expecting test in this file (none of which
# set `Accept` at all) still passing unchanged.
# --------------------------------------------------------------------


def test_expired_session_plain_browser_post_redirects_to_login(client: TestClient) -> None:
    r = client.post(
        "/conversations",
        headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/login?next=/conversations"


def test_expired_session_api_caller_still_gets_plain_401_json(client: TestClient) -> None:
    r = client.post("/conversations", follow_redirects=False)
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/json")


def test_expired_session_htmx_fragment_still_gets_plain_401_json_even_with_html_accept(
    client: TestClient,
) -> None:
    r = client.post(
        "/conversations/1/archive",
        headers={"Accept": "text/html", "HX-Request": "true"},
        follow_redirects=False,
    )
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/json")


def test_login_next_param_round_trips_and_redirects_there_after_login(
    client: TestClient,
) -> None:
    r = client.get("/login?next=%2Fnotes", follow_redirects=False)
    assert r.status_code == 200
    assert 'name="next" value="/notes"' in r.text

    r2 = client.post(
        "/login",
        data={"org_key": ORG_KEY, "editor": "Nexter", "next": "/notes"},
        follow_redirects=False,
    )
    assert r2.status_code == 303
    assert r2.headers["location"] == "/notes"


def test_login_next_param_rejects_an_absolute_url_open_redirect(client: TestClient) -> None:
    r = client.post(
        "/login",
        data={
            "org_key": ORG_KEY,
            "editor": "Nexter Two",
            "next": "https://evil.example/steal-session",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/"


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 5: `GET /c/abc` (a non-integer path
# parameter) used to fall through to FastAPI's own raw
# `{"detail":[{"type":"int_parsing",...}]}` validation JSON, bypassing
# the branded `error.html` punch item 11 exists for.
# --------------------------------------------------------------------


def test_invalid_path_param_renders_branded_page_not_raw_422(
    client: TestClient, auth_cookies
) -> None:
    r = client.get("/c/not-a-number", cookies=auth_cookies)
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("text/html")
    assert '"type":"int_parsing"' not in r.text
    assert '"detail"' not in r.text


# --------------------------------------------------------------------
# App checklist 2026-09-18, P0 item 6: confirm the RequestValidationError
# handler `_register_error_handlers` (`app.py`) added for the finding-5
# `/c/not-a-number` case above -- credited there to "the security fixer"
# -- also covers every OTHER query-param/path-param validation error a
# person can reach by editing the address bar, on the pages the web-pages
# bug hunt covered (`bughunt-2026-09-18-web-pages.md` finding 12, which
# named `/notes?page=abc` and `/coverage?days=abc` specifically; `/review`
# and `/notes/{slug}/edit` carry the same `int` query-param shape and are
# checked here too). Every one of these already passes against this
# checkout -- `_validation_exception_handler` is a blanket handler, not a
# per-route allowlist -- so this locks that in as a regression test
# rather than a fix. Atlas (`routes/atlas.py`) carries no typed int/path
# param at all (`q`/`active`/`slug` are all bare `str`, hand-validated
# with `bounded_str` inside the route body, which raises a plain
# `HTTPException(400, ...)` -- already routed through `error.html` by the
# OTHER handler, `_http_exception_handler`) -- there is no FastAPI-level
# RequestValidationError case to reach there, confirmed by reading every
# route signature in that module rather than assumed.
# --------------------------------------------------------------------


def test_notes_index_invalid_page_renders_branded_page_not_raw_422(
    client: TestClient, auth_cookies
) -> None:
    r = client.get("/notes?page=abc", cookies=auth_cookies)
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("text/html")
    assert '"type":"int_parsing"' not in r.text
    assert '"detail"' not in r.text


def test_coverage_invalid_days_renders_branded_page_not_raw_422(
    client: TestClient, auth_cookies
) -> None:
    r = client.get("/coverage?days=abc", cookies=auth_cookies)
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("text/html")
    assert '"type":"int_parsing"' not in r.text
    assert '"detail"' not in r.text


def test_review_invalid_page_renders_branded_page_not_raw_422(
    client: TestClient, auth_cookies
) -> None:
    r = client.get("/review?page=abc", cookies=auth_cookies)
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("text/html")
    assert '"type":"int_parsing"' not in r.text
    assert '"detail"' not in r.text


def test_note_edit_form_invalid_claim_id_renders_branded_page_not_raw_422(
    client: TestClient, auth_cookies, any_claim
) -> None:
    note_slug, *_ = any_claim
    r = client.get(f"/notes/{note_slug}/edit?claim_id=abc", cookies=auth_cookies)
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("text/html")
    assert '"type":"int_parsing"' not in r.text
    assert '"detail"' not in r.text


def test_favicon_and_meta_description_present(client: TestClient) -> None:
    """Fix wave 2026-09-15 (v3): icon.svg's own leading comment used to
    contain a literal double hyphen three times, which XML forbids inside
    comments -- `xmllint --noout` failed on it and Chrome silently fell
    back to the generic globe tab icon instead of parsing the file at
    all. The `?v=2` on the SVG link busts that cached failure; the PNG
    links are fallbacks for contexts that don't read an SVG favicon
    (Apple's home-screen icon, some pinned-tab UIs). Bumped to `?v=3`
    2026-09-18 when the mark recolored from ink to rhumb-line red -- same
    per-origin favicon cache concern, now on all three links since the
    PNGs were regenerated too."""
    r = client.get("/login")
    assert r.status_code == 200
    assert 'rel="icon" type="image/svg+xml" href="/static/icon.svg?v=3"' in r.text
    assert 'rel="icon" type="image/png" sizes="32x32" href="/static/icon-32.png?v=3"' in r.text
    assert 'rel="apple-touch-icon" href="/static/apple-touch-icon.png?v=3"' in r.text
    assert '<meta name="description"' in r.text

    icon = client.get("/static/icon.svg")
    assert icon.status_code == 200
    # The actual regression: this must be well-formed XML, or a browser's
    # SVG favicon parser rejects it the same way `xmllint --noout` would.
    ElementTree.fromstring(icon.content)

    icon_32 = client.get("/static/icon-32.png")
    assert icon_32.status_code == 200
    assert icon_32.headers["content-type"] == "image/png"
    assert icon_32.content[:8] == b"\x89PNG\r\n\x1a\n"

    apple_icon = client.get("/static/apple-touch-icon.png")
    assert apple_icon.status_code == 200
    assert apple_icon.headers["content-type"] == "image/png"


def test_shared_escape_html_script_is_loaded_on_every_page_and_served(
    client: TestClient, auth_cookies
) -> None:
    """Review finding 22 (grouped nits): chat.html and graph.html each
    used to define their own escapeHtml -- now both read
    `window.PortolanEscapeHtml` from one shared, same-origin static file
    base.html loads on every page."""
    r = client.get("/login")
    assert r.status_code == 200
    assert '<script src="/static/js/escape-html.js"></script>' in r.text

    asset = client.get("/static/js/escape-html.js")
    assert asset.status_code == 200
    assert "PortolanEscapeHtml" in asset.text

    chat_page = client.get("/", cookies=auth_cookies)
    assert "window.PortolanEscapeHtml" in chat_page.text
    assert "function escapeHtml(s) {" not in chat_page.text

    graph_page = client.get("/graph", cookies=auth_cookies)
    assert "window.PortolanEscapeHtml" in graph_page.text
    assert "function escapeHtml(s) {" not in graph_page.text


# --------------------------------------------------------------------
# Punch item 12: a real /notes index, so the note back-link works.
# --------------------------------------------------------------------


def test_note_back_link_no_longer_404s(client: TestClient, auth_cookies, any_claim) -> None:
    note_slug, *_ = any_claim
    note_page = client.get(f"/notes/{note_slug}", cookies=auth_cookies)
    assert 'href="/notes"' in note_page.text

    r = client.get("/notes", cookies=auth_cookies)
    assert r.status_code == 200


def test_notes_index_lists_a_known_note_and_filters_by_type(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, any_claim
) -> None:
    note_slug, *_ = any_claim
    row = _one(demo_conn, "SELECT title, type FROM notes WHERE slug = %s", (note_slug,))
    assert row is not None
    title, note_type = row

    r = client.get("/notes", cookies=auth_cookies)
    assert r.status_code == 200
    assert f'href="/notes/{note_slug}"' in r.text

    matching = client.get("/notes", params={"type": note_type}, cookies=auth_cookies)
    assert matching.status_code == 200
    assert f'href="/notes/{note_slug}"' in matching.text

    other_type = "competitor" if note_type != "competitor" else "product"
    filtered_out = client.get("/notes", params={"type": other_type}, cookies=auth_cookies)
    assert filtered_out.status_code == 200
    assert f'href="/notes/{note_slug}"' not in filtered_out.text

    searched = client.get("/notes", params={"q": title[:6]}, cookies=auth_cookies)
    assert searched.status_code == 200
    assert f'href="/notes/{note_slug}"' in searched.text


def test_notes_index_requires_session(client: TestClient) -> None:
    r = client.get("/notes", follow_redirects=False)
    assert r.status_code == 302


def test_notes_index_paginates_at_50(client: TestClient, demo_conn: psycopg.Connection) -> None:
    from portolan.web.queries import NOTES_INDEX_PAGE_SIZE, list_notes

    assert NOTES_INDEX_PAGE_SIZE == 50
    rows, total = list_notes(demo_conn, page=1, page_size=50)
    assert len(rows) <= 50
    assert total >= len(rows)


def test_notes_index_search_treats_percent_and_underscore_literally(
    demo_conn: psycopg.Connection,
) -> None:
    """`/notes?q=%` must find notes that contain a `%`, not every note."""
    from portolan.web.queries import list_notes

    with demo_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM notes")
        (all_notes,) = cur.fetchone()
        cur.execute(
            "SELECT count(*) FROM notes WHERE strpos(title, '%%') > 0 OR strpos(slug, '%%') > 0"
        )
        (with_percent,) = cur.fetchone()
        cur.execute(
            "SELECT count(*) FROM notes WHERE strpos(title, '_') > 0 OR strpos(slug, '_') > 0"
        )
        (with_underscore,) = cur.fetchone()
    assert with_percent < all_notes and with_underscore < all_notes

    assert list_notes(demo_conn, q="%")[1] == with_percent
    assert list_notes(demo_conn, q="_")[1] == with_underscore


# --------------------------------------------------------------------
# Punch item 13: the review queue never shows a raw dict repr.
# --------------------------------------------------------------------


def test_review_queue_renders_payload_as_definition_list_not_a_dict_repr(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('other', %s::jsonb) RETURNING id",
            ('{"note_slug": "f1", "reasons": ["numeral_not_supplied"]}',),
        )
        (row_id,) = cur.fetchone()
    demo_conn.commit()

    r = client.get("/review", cookies=auth_cookies)
    assert r.status_code == 200
    assert "{'note_slug'" not in r.text
    assert '{"note_slug"' not in r.text
    assert 'class="payload-dl"' in r.text
    assert "<dt>note_slug</dt>" in r.text
    assert "<dd>f1</dd>" in r.text
    del row_id  # only needed to force the insert; not asserted on directly


def test_review_queue_ambiguous_entity_reads_in_plain_english(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies
) -> None:
    """Review-queue-usability fix, 2026-09-18: a `source_slug IS NULL`
    `ambiguous_entity` row only ever comes from the CRM-dedup pass
    (`portolan.registry.importer`), never a document mention -- the page
    now says so in plain English (`portolan.web.review_presenter`)
    instead of a raw "? ambiguous_entity" badge, and still shows both
    candidate names (or their raw id, when neither resolves to a real
    registry entity, as here)."""
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) VALUES "
            "('ambiguous_entity', %s::jsonb, NULL)",
            ('{"mention": "Ashgrove", "candidates": ["ashgrove-health", "ashgrove-medical"]}',),
        )
    demo_conn.commit()

    r = client.get("/review", cookies=auth_cookies)
    assert r.status_code == 200
    assert "Ambiguous name" in r.text  # the plain-word category badge
    assert "importing the CRM" in r.text
    assert "Ashgrove" in r.text
    assert "ashgrove-health" in r.text
    assert "ashgrove-medical" in r.text


def test_review_row_links_to_the_original_at_its_source(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, call_source
) -> None:
    slug, _locator = call_source
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) VALUES "
            "('unknown_entity', %s::jsonb, %s)",
            ('{"mention": "a brand new name", "context": {"fact_id": "f9"}}', slug),
        )
    demo_conn.commit()

    r = client.get("/review", cookies=auth_cookies)
    assert r.status_code == 200
    assert f'href="/originals/{slug}"' in r.text or f"/originals/{slug}?" in r.text


def test_review_page_survives_a_source_whose_original_view_cannot_be_built(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, store_root: Path
) -> None:
    """`context_snippet` propagates `_build_view` errors; an `edit` source
    (allowed by the schema, not a `SourceType`) with a stored original
    raises there. The card renders without its snippet instead of the
    whole queue returning a 500."""
    slug = "S-edit-review-snippet-test"
    s3_key = f"originals/edit/{slug}"
    LocalStore(store_root).put(s3_key, b"an edited sentence", "text/plain")
    fact = {
        "id": "F-edit-snippet",
        "type": "complaint",
        "entities": ["Nobody In Particular"],
        "claim": "An unsupported quote on an edit source.",
        "quote": "no such words",
        "locator": "L1",
        "speaker": None,
        "valid_from": None,
        "valid_to": None,
        "severity": None,
        "sentiment": -1,
        "confidence": 0.5,
    }
    with demo_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, source_date)
            VALUES (%s, 'edit', %s, '2026-03-02')
            ON CONFLICT (slug) DO UPDATE SET s3_key = EXCLUDED.s3_key
            """,
            (slug, s3_key),
        )
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) "
            "VALUES ('quote_unsupported', %s::jsonb, %s) RETURNING id",
            (json.dumps({"reason": "quote not found", "fact": fact}), slug),
        )
        (row_id,) = cur.fetchone()
    demo_conn.commit()
    try:
        r = client.get("/review?kind=Unsupported quote", cookies=auth_cookies)
        assert r.status_code == 200
        assert f"/review/{row_id}/resolve" in r.text
    finally:
        with demo_conn.cursor() as cur:
            cur.execute("UPDATE review_queue SET status = 'dismissed' WHERE id = %s", (row_id,))
        demo_conn.commit()


# --------------------------------------------------------------------
# Review-queue-usability follow-up, 2026-09-18: render-time evidence
# backfill for a row written before portolan.merge.resolve saved
# claim/quote/locator onto the payload -- "every row on the live
# Railway database" carries only {mention, context: {fact_id,
# fact_type}}, the exact pre-fix shape reused below.
# --------------------------------------------------------------------


def test_review_row_with_a_legacy_payload_backfills_the_evidence_from_a_matching_claim(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, call_source
) -> None:
    """A legacy row has no locator of its own to look a claim up by
    (that's the whole point of the follow-up: the old payload shape
    saved none) -- recovery falls back to a same-source claim whose text
    names the same mention (`portolan.web.queries.
    backfill_fact_evidence`'s tier 2)."""
    slug, locator = call_source
    with demo_conn.cursor() as cur:
        cur.execute("SELECT note_slug FROM claims WHERE source_slug = %s LIMIT 1", (slug,))
        note_slug = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, quote) "
            "VALUES (%s, 'risk', %s, %s, %s, %s) RETURNING id",
            (
                note_slug,
                "Waverunner Regional said the Marlowe rollout keeps slipping.",
                slug,
                "backfill-fixture#1",
                "The Marlowe rollout keeps slipping every single week.",
            ),
        )
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) VALUES "
            "('unknown_entity', %s::jsonb, %s)",
            ('{"mention": "Marlowe", "context": {"fact_id": "f7", "fact_type": "risk"}}', slug),
        )
    demo_conn.commit()
    del locator  # only needed to force call_source's own fixture join; not used directly

    r = client.get("/review", cookies=auth_cookies)
    assert r.status_code == 200
    # The recovered claim sentence and its verbatim quote (the blockquote)
    # both render -- other rows elsewhere on the (session-shared) page may
    # legitimately show the "couldn't find it" note themselves, so this
    # only checks that THIS row's own evidence made it onto the page, not
    # that the phrase is absent globally.
    assert "Waverunner Regional said the Marlowe rollout keeps slipping." in r.text
    assert "The Marlowe rollout keeps slipping every single week." in r.text  # the blockquote


def test_review_row_with_a_legacy_payload_and_no_matching_claim_shows_the_fallback_note(
    client: TestClient, demo_conn: psycopg.Connection, auth_cookies, call_source
) -> None:
    """When neither backfill tier finds anything, the card says so in
    plain language rather than rendering an empty evidence block."""
    slug, _locator = call_source
    with demo_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) VALUES "
            "('unknown_entity', %s::jsonb, %s)",
            (
                '{"mention": "Zorbaxion Nineteen", "context": '
                '{"fact_id": "f8", "fact_type": "risk"}}',
                slug,
            ),
        )
    demo_conn.commit()

    r = client.get("/review", cookies=auth_cookies)
    assert r.status_code == 200
    assert "Portolan no longer has the exact sentence for this mention" in r.text


def test_backfill_fact_evidence_prefers_an_exact_locator_match(
    demo_conn: psycopg.Connection, call_source
) -> None:
    from portolan.web.queries import backfill_fact_evidence

    slug, locator = call_source
    result = backfill_fact_evidence(demo_conn, source_slug=slug, mention=None, locator=locator)
    assert result is not None
    assert result["locator"] == locator
    assert result["claim"]


def test_backfill_fact_evidence_returns_none_when_nothing_matches(
    demo_conn: psycopg.Connection, call_source
) -> None:
    from portolan.web.queries import backfill_fact_evidence

    slug, _locator = call_source
    result = backfill_fact_evidence(
        demo_conn, source_slug=slug, mention="Definitely Not A Real Mention Zzq", locator=None
    )
    assert result is None


# --------------------------------------------------------------------
# Punch item 15: a source page hands you claims and their locators.
# --------------------------------------------------------------------


def test_source_page_lists_citing_claims_with_dig_chips(
    client: TestClient, auth_cookies, any_claim
) -> None:
    note_slug, _claim_id, source_slug, locator = any_claim
    r = client.get(f"/sources/{source_slug}", cookies=auth_cookies)
    assert r.status_code == 200
    assert "Cited from" in r.text
    assert f'href="/notes/{note_slug}"' in r.text
    assert f"@ {locator}" in r.text
    assert 'class="btn chip dig"' in r.text
    # the hash, if present, is truncated with the full value in title=
    if "sha256:" in r.text:
        match = re.search(r'title="([0-9a-f]{64})">sha256:', r.text)
        assert match is not None, "full sha256 should be recoverable from title="


# --------------------------------------------------------------------
# Punch item 14: the graph is labelled and has an empty-state message.
# --------------------------------------------------------------------


def test_graph_page_ships_node_labels_and_empty_state(client: TestClient, auth_cookies) -> None:
    """Fix wave 2026-09-12: the 2D island moved from hand-drawn SVG `<g
    class="node">` elements to a force-graph canvas, so per-node labels
    are painted on the canvas (`nodeCanvasObject`) rather than being
    separate DOM elements. Restyle wave 2026-09-15
    (graph-reference-aicodingdictionary.md, build instructions "Labels"):
    labels are now always visible, not hover-only -- the old
    `labelsAlwaysOn` (<=40 nodes) threshold from the UX eval's punch item
    14 is gone; opacity follows node size instead of a node-count cutoff."""
    r = client.get("/graph", cookies=auth_cookies)
    assert r.status_code == 200
    assert "labelsAlwaysOn" not in r.text
    assert "function labelOpacity(val)" in r.text
    assert 'id="graph-empty"' in r.text


@pytest.fixture(autouse=True)
def _single_chat_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests script one model call per answer; agent mode is covered in
    # tests/test_chat_agent.py.
    monkeypatch.setenv("PORTOLAN_CHAT_MODE", "single")
