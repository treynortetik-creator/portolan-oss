"""Web sign-in with WorkOS AuthKit (`portolan.web.signin`, `routes/login.py`,
migration 0038): every route but the public ones needs a session, forged and
cross-mode cookies are refused, the organization and domain policy, the verified
identity on everything a person does, the viewer role and admin pages, logout,
the MCP door left alone, and web_auth="none" unchanged.

A fake provider stands in for WorkOS; one test drives the real SDK's URL
builders, which make no network call.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import jwt
import psycopg
import pytest
from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner, URLSafeTimedSerializer

from portolan.activity import record_activity
from portolan.config import Settings
from portolan.llm.fake import FakeLLMClient
from portolan.merge.actions import merge_action
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.web import answer as answer_module
from portolan.web import signin
from portolan.web.app import create_app
from portolan.web.auth import (
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_SALT,
    SESSION_MAX_AGE_SECONDS,
    WORKOS_SESSION_COOKIE_SALT,
    create_session_cookie,
)
from portolan.web.deps import get_conn, get_embedder_dep, get_llm_dep
from portolan.web.routes import chat as chat_routes
from tests.merge_helpers import (
    ASHGROVE_MEDICAL,
    build_action,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

SESSION_SECRET = "test-session-secret-for-web-signin-not-a-real-secret"
ORG_KEY = "test-org-key-for-web-signin"
ORG = "org_01PILOTTEST"
BASE = "http://testserver"
FAKE_AUTHKIT = "https://fake-authkit.example.test"
MCP_KEY = "static-mcp-key-for-web-signin-tests-0123456789abcdef"
ADMIN = "ada@pilot.example"


def _person(email: str = "sam@pilot.example", **overrides: Any) -> signin.SignedInPerson:
    values: dict[str, Any] = {
        "user_id": f"user_{email.split('@')[0]}",
        "email": email,
        "email_verified": True,
        "first_name": "Sam",
        "last_name": "Lee",
        "organization_id": ORG,
        "session_id": f"session_{email.split('@')[0]}",
    }
    values.update(overrides)
    return signin.SignedInPerson(**values)


class FakeWorkOS:
    """WorkOS as the app sees it: an authorize URL, a code exchange that checks
    PKCE, and a logout URL. `issue` decides who the next code signs in."""

    def __init__(self) -> None:
        self.challenges: dict[str, str] = {}
        self.codes: dict[str, tuple[signin.SignedInPerson, str]] = {}
        self.exchanges: list[dict[str, Any]] = []

    def authorization_url(self, *, redirect_uri: str, state: str, code_challenge: str) -> str:
        self.challenges[state] = code_challenge
        return f"{FAKE_AUTHKIT}/authorize?state={state}&redirect_uri={redirect_uri}"

    def issue(self, state: str, person: signin.SignedInPerson) -> str:
        code = f"code_{len(self.codes)}"
        self.codes[code] = (person, self.challenges[state])
        return code

    def authenticate(
        self, *, code: str, code_verifier: str, organization_id: str | None
    ) -> signin.SignedInPerson:
        self.exchanges.append({"code": code, "organization_id": organization_id})
        if code not in self.codes:
            raise signin.SignInError("workos_BadRequestError")
        person, challenge = self.codes.pop(code)
        digest = hashlib.sha256(code_verifier.encode()).digest()
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != challenge:
            raise signin.SignInError("workos_BadRequestError")
        return person

    def logout_url(self, *, session_id: str, return_to: str | None) -> str:
        return f"{FAKE_AUTHKIT}/logout?session_id={session_id}&return_to={return_to}"


def _settings(db_url: str, tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": db_url,
        "tenant": "test",
        "store_root": tmp_path / "store",
        "llm_provider": "fake",
        "embed_provider": "fake",
        "session_secret": SESSION_SECRET,
        "org_key": ORG_KEY,
        "cookie_secure": False,
        "public_base_url": BASE,
        "web_auth": "workos",
        "workos_client_id": "client_01TEST",
        "workos_api_key": "sk_test_not_a_real_key_0123456789",
        "web_auth_org_id": ORG,
        "admin_emails": ADMIN,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _app(db, db_url: str, tmp_path: Path, fake: FakeWorkOS | None = None, **overrides: Any):
    app = create_app(_settings(db_url, tmp_path, **overrides), workos_provider=fake or FakeWorkOS())
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([])
    return app


@pytest.fixture()
def fake() -> FakeWorkOS:
    return FakeWorkOS()


@pytest.fixture()
def client(db: psycopg.Connection, db_url: str, tmp_path: Path, fake: FakeWorkOS):
    with TestClient(_app(db, db_url, tmp_path, fake)) as c:
        yield c


def _start(client: TestClient, next_path: str | None = None) -> str:
    url = "/login" + (f"?next={next_path}" if next_path else "")
    resp = client.get(url, follow_redirects=False)
    assert resp.status_code == 303
    location = resp.headers["location"]
    assert location.startswith(f"{FAKE_AUTHKIT}/authorize?")
    return parse_qs(urlparse(location).query)["state"][0]


def _sign_in(client: TestClient, fake: FakeWorkOS, person: signin.SignedInPerson, next_path=None):
    state = _start(client, next_path)
    return client.get(
        f"/auth/callback?code={fake.issue(state, person)}&state={state}", follow_redirects=False
    )


def _signed_in(client, fake, person=None) -> str:
    """Sign `person` in on `client` and return their session cookie."""
    client.cookies.clear()
    resp = _sign_in(client, fake, person or _person())
    assert resp.status_code == 303, resp.text
    return resp.cookies[SESSION_COOKIE_NAME]


def _use(client: TestClient, cookie: str) -> None:
    # The domain the TestClient's jar files the app's own cookies under, so a
    # response that deletes the cookie deletes this one.
    client.cookies.clear()
    client.cookies.set(SESSION_COOKIE_NAME, cookie, domain="testserver.local")


def _sample(path: str) -> str:
    """A concrete URL for a route template: ids become 1, anything else x."""
    return re.sub(r"\{(\w+)(:\w+)?\}", lambda m: "1" if m.group(1).endswith("id") else "x", path)


# ---------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------


def test_every_route_but_the_public_ones_sends_a_signed_out_browser_to_login(client) -> None:
    # The OpenAPI table lists every include_router route (the installed FastAPI wraps
    # `app.routes` in `_IncludedRouter`); the paths hidden from it are added by hand,
    # and a path nobody registered shows the gate denies by default.
    table = [
        (path, method.upper())
        for path, operations in client.app.openapi()["paths"].items()
        for method in operations
    ]
    table += [("/how-it-works", "GET"), ("/no-such-page", "GET"), ("/no-such-api", "POST")]
    checked = 0
    for template, method in table:
        path = _sample(template)
        if path in signin.PUBLIC_PATHS or path.startswith(signin.PUBLIC_PREFIXES):
            continue
        if method == "GET":
            resp = client.get(path, headers={"Accept": "text/html"}, follow_redirects=False)
            assert resp.status_code == 303, (method, template, resp.status_code)
            assert resp.headers["location"] == f"/login?next={path}", template
        else:
            resp = client.request(method, path, follow_redirects=False)
            assert resp.status_code == 401, (method, template, resp.status_code)
        checked += 1
    # Open without a login under web_auth="none"; gated here.
    for path in ("/how-it-works", "/search?q=x"):
        assert client.get(path, follow_redirects=False).status_code == 401, path
    assert checked > 40


def test_the_public_paths_answer_without_a_session(client) -> None:
    assert client.get("/healthz").status_code == 200
    assert client.get("/static/does-not-exist.css").status_code == 404
    assert client.get("/signed-out").status_code == 200
    assert client.get("/login", follow_redirects=False).status_code == 303


def test_an_api_caller_without_a_session_gets_401_json_not_a_redirect(client) -> None:
    resp = client.get("/graph.json", follow_redirects=False)
    assert resp.status_code == 401
    assert resp.json() == {"detail": "login required"}
    htmx = client.get(
        "/atlas/panel",
        headers={"Accept": "text/html", "HX-Request": "true"},
        follow_redirects=False,
    )
    assert htmx.status_code == 401


def test_the_login_redirect_keeps_where_the_person_was_going(client, fake) -> None:
    gated = client.get("/atlas", headers={"Accept": "text/html"}, follow_redirects=False)
    assert gated.headers["location"] == "/login?next=/atlas"
    resp = _sign_in(client, fake, _person(), next_path="/atlas")
    assert resp.headers["location"] == "/atlas"


# ---------------------------------------------------------------------
# The flow and forged cookies
# ---------------------------------------------------------------------


def test_sign_in_sets_a_signed_httponly_secure_lax_cookie_with_an_expiry(
    db, db_url, tmp_path, fake
) -> None:
    with TestClient(_app(db, db_url, tmp_path, fake, cookie_secure=True)) as c:
        start = c.get("/login", follow_redirects=False)
        flow_cookie = start.headers["set-cookie"].lower()
        for attr in (
            "httponly",
            "secure",
            "samesite=lax",
            f"max-age={signin.FLOW_MAX_AGE_SECONDS}",
        ):
            assert attr in flow_cookie
        state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
        resp = c.get(
            f"/auth/callback?code={fake.issue(state, _person())}&state={state}",
            cookies={signin.FLOW_COOKIE_NAME: start.cookies[signin.FLOW_COOKIE_NAME]},
            follow_redirects=False,
        )
    assert resp.status_code == 303
    session_cookie = next(
        h for h in resp.headers.get_list("set-cookie") if h.startswith(SESSION_COOKIE_NAME)
    ).lower()
    for attr in ("httponly", "secure", "samesite=lax", f"max-age={SESSION_MAX_AGE_SECONDS}"):
        assert attr in session_cookie
    assert fake.exchanges == [{"code": "code_0", "organization_id": ORG}]


def test_the_callback_refuses_a_state_that_does_not_match_the_flow_cookie(client, fake) -> None:
    real_state = _start(client)
    code = fake.issue(real_state, _person())
    resp = client.get(f"/auth/callback?code={code}&state=forged-state", follow_redirects=False)
    assert resp.status_code == 403
    assert SESSION_COOKIE_NAME not in resp.cookies
    assert fake.exchanges == []


def test_the_callback_refuses_with_no_flow_cookie(client, fake) -> None:
    state = _start(client)
    code = fake.issue(state, _person())
    client.cookies.clear()
    resp = client.get(f"/auth/callback?code={code}&state={state}", follow_redirects=False)
    assert resp.status_code == 403
    assert fake.exchanges == []


def test_the_callback_refuses_an_error_from_workos(client) -> None:
    state = _start(client)
    resp = client.get(
        f"/auth/callback?error=access_denied&error_description=no&state={state}",
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert SESSION_COOKIE_NAME not in resp.cookies


@pytest.mark.parametrize(
    "forge",
    [
        pytest.param(
            lambda: URLSafeTimedSerializer("guessed", salt=WORKOS_SESSION_COOKIE_SALT).dumps(
                {"email": "attacker@pilot.example", "name": "A"}
            ),
            id="wrong-secret",
        ),
        pytest.param(
            lambda: create_session_cookie("the maintainer", secret=SESSION_SECRET),
            id="a-shared-key-login-cookie",
        ),
        pytest.param(
            lambda: URLSafeTimedSerializer(SESSION_SECRET, salt=SESSION_COOKIE_SALT).dumps(
                {"editor": "x", "email": "attacker@pilot.example"}
            ),
            id="right-secret-other-salt",
        ),
        pytest.param(
            lambda: URLSafeTimedSerializer(SESSION_SECRET, salt=WORKOS_SESSION_COOKIE_SALT).dumps(
                {"name": "No Email"}
            ),
            id="no-email",
        ),
    ],
)
def test_a_forged_cookie_is_rejected(client, forge) -> None:
    cookie = forge()
    _use(client, cookie)
    assert (
        client.get("/", headers={"Accept": "text/html"}, follow_redirects=False).status_code == 303
    )
    assert client.get("/graph.json", follow_redirects=False).status_code == 401


def test_a_tampered_or_expired_cookie_is_rejected(client, fake) -> None:
    _signed_in(client, fake)
    good = client.cookies[SESSION_COOKIE_NAME]
    assert client.get("/graph.json").status_code == 200
    tampered = good[:-2] + ("A" if good[-2] != "A" else "B") + good[-1]
    _use(client, tampered)
    assert client.get("/graph.json", follow_redirects=False).status_code == 401

    class _OldSigner(TimestampSigner):
        def get_timestamp(self) -> int:
            return super().get_timestamp() - SESSION_MAX_AGE_SECONDS - 60

    old = URLSafeTimedSerializer(SESSION_SECRET, salt=WORKOS_SESSION_COOKIE_SALT, signer=_OldSigner)
    expired = old.dumps({"email": "sam@pilot.example", "name": "Sam Lee", "sid": "s"})
    _use(client, expired)
    assert client.get("/graph.json", follow_redirects=False).status_code == 401


def test_the_org_key_form_sets_no_session_under_workos(client) -> None:
    resp = client.post(
        "/login", data={"org_key": ORG_KEY, "editor": "Mallory"}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"
    assert SESSION_COOKIE_NAME not in resp.cookies


# ---------------------------------------------------------------------
# Organization and domain policy
# ---------------------------------------------------------------------


def test_someone_outside_the_organization_is_refused(client, fake, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        resp = _sign_in(client, fake, _person("eve@pilot.example", organization_id="org_OTHER"))
    assert resp.status_code == 403
    assert SESSION_COOKIE_NAME not in resp.cookies
    assert "a member of this Portolan" in resp.text
    assert "session_id=session_eve" in resp.text  # "sign out and use another account"
    assert "web sign-in refused: reason=organization" in caplog.text
    assert "eve@pilot.example" not in caplog.text
    none_org = _sign_in(client, fake, _person("nora@pilot.example", organization_id=None))
    assert none_org.status_code == 403


@pytest.mark.parametrize(
    ("person", "allowed"),
    [
        (_person("sam@pilot.example"), True),
        (_person("sam@PILOT.example"), True),
        (_person("sam@pilot.example.evil.test"), False),
        (_person("sam@evil.test"), False),
        (_person("sam@pilot.example", email_verified=False), False),
    ],
)
def test_the_allowed_domain_check(db, db_url, tmp_path, fake, person, allowed) -> None:
    app = _app(
        db,
        db_url,
        tmp_path,
        fake,
        web_auth_org_id=None,
        web_auth_allowed_domains="pilot.example, other.example",
    )
    with TestClient(app) as c:
        resp = _sign_in(c, fake, replace(person, organization_id=None))
    assert (resp.status_code == 303) is allowed
    assert (SESSION_COOKIE_NAME in resp.cookies) is allowed


def test_organization_and_domain_both_apply_when_both_are_set(db, db_url, tmp_path, fake) -> None:
    app = _app(db, db_url, tmp_path, fake, web_auth_allowed_domains="pilot.example")
    with TestClient(app) as c:
        assert _sign_in(c, fake, _person("sam@evil.test")).status_code == 403
        assert _sign_in(c, fake, _person(organization_id="org_OTHER")).status_code == 403
        assert _sign_in(c, fake, _person()).status_code == 303


def test_the_organization_defaults_to_the_mcp_oauth_organization(
    db, db_url, tmp_path, fake
) -> None:
    app = _app(db, db_url, tmp_path, fake, web_auth_org_id=None, mcp_oauth_org_id=ORG)
    with TestClient(app) as c:
        assert _sign_in(c, fake, _person(organization_id="org_OTHER")).status_code == 403
        assert _sign_in(c, fake, _person()).status_code == 303


@pytest.mark.parametrize(
    ("overrides", "names"),
    [
        ({"workos_client_id": None}, ["PORTOLAN_WORKOS_CLIENT_ID"]),
        ({"workos_api_key": " "}, ["PORTOLAN_WORKOS_API_KEY"]),
        ({"public_base_url": None}, ["PORTOLAN_PUBLIC_URL"]),
        ({"web_auth_org_id": None}, ["PORTOLAN_WEB_AUTH_ALLOWED_DOMAINS"]),
        ({"session_secret": None}, ["PORTOLAN_SESSION_SECRET"]),
    ],
)
def test_the_app_refuses_to_start_with_workos_half_configured(
    db_url, tmp_path, overrides, names
) -> None:
    with pytest.raises(RuntimeError) as exc:
        create_app(_settings(db_url, tmp_path, **overrides), workos_provider=FakeWorkOS())
    for name in names:
        assert name in str(exc.value)


def test_the_api_key_is_a_secret_and_never_in_the_sign_in_url(db_url, tmp_path) -> None:
    settings = _settings(db_url, tmp_path)
    assert "sk_test_not_a_real_key" not in repr(settings)
    assert "sk_test_not_a_real_key" not in json.dumps(settings.model_dump(mode="json"))
    provider = signin.build_provider(settings)  # the real SDK; building URLs is local
    url = provider.authorization_url(
        redirect_uri=signin.redirect_uri(settings), state="st", code_challenge="ch"
    )
    query = parse_qs(urlparse(url).query)
    assert url.startswith("https://api.workos.com/user_management/authorize?")
    assert query["client_id"] == ["client_01TEST"]
    assert query["provider"] == ["authkit"]
    assert query["redirect_uri"] == [f"{BASE}/auth/callback"]
    assert query["code_challenge_method"] == ["S256"]
    assert "sk_test_not_a_real_key" not in url
    logout = provider.logout_url(session_id="session_1", return_to=f"{BASE}/signed-out")
    assert parse_qs(urlparse(logout).query) == {
        "session_id": ["session_1"],
        "return_to": [f"{BASE}/signed-out"],
    }


def test_the_sdk_provider_picks_our_organization_for_a_member_of_several(
    db_url, tmp_path, monkeypatch
) -> None:
    from workos._errors import OrganizationSelectionRequiredError
    from workos.user_management.models import AuthenticateResponse

    provider = signin.build_provider(_settings(db_url, tmp_path))
    um = provider._client.user_management
    token = jwt.encode({"sid": "session_77"}, "not-checked", algorithm="HS256")
    response = AuthenticateResponse.from_dict(
        {
            "user": {
                "id": "user_1",
                "first_name": "Sam",
                "last_name": "Lee",
                "profile_picture_url": None,
                "email": "sam@pilot.example",
                "email_verified": True,
                "external_id": None,
                "last_sign_in_at": None,
                "created_at": "2026-09-25T00:00:00Z",
                "updated_at": "2026-09-25T00:00:00Z",
            },
            "access_token": token,
            "refresh_token": "r",
            "organization_id": ORG,
        }
    )
    selected: list[str] = []

    def _code(**_kwargs):
        raise OrganizationSelectionRequiredError(
            "pick one", pending_authentication_token="pending_1"
        )

    def _select(*, pending_authentication_token: str, organization_id: str):
        selected.append(f"{pending_authentication_token}:{organization_id}")
        return response

    monkeypatch.setattr(um, "authenticate_with_code", _code)
    monkeypatch.setattr(um, "authenticate_with_organization_selection", _select)
    person = provider.authenticate(code="c", code_verifier="v", organization_id=ORG)
    assert selected == [f"pending_1:{ORG}"]
    assert (person.email, person.name, person.organization_id, person.session_id) == (
        "sam@pilot.example",
        "Sam Lee",
        ORG,
        "session_77",
    )
    with pytest.raises(signin.SignInError) as exc:
        provider.authenticate(code="c", code_verifier="v", organization_id=None)
    assert exc.value.reason == "organization_selection"


# ---------------------------------------------------------------------
# Identity everywhere, roles, admin
# ---------------------------------------------------------------------


IDENTITY = "Sam Lee <sam@pilot.example>"


def _cited_answer(**_kwargs) -> answer_module.Answer:
    return answer_module.Answer(
        sentences=[answer_module.Sentence(text="They asked about the renewal.", citations=[])],
        retrieved=[],
        control_ok=True,
        usage={},
    )


def test_the_verified_identity_is_on_chat_review_and_action_records(
    client, fake, db, monkeypatch, loaded_registry: Registry
) -> None:
    _signed_in(client, fake)
    page = client.get("/")
    assert "Sam Lee" in page.text

    monkeypatch.setattr(chat_routes, "answer_question", lambda *a, **k: _cited_answer(**k))
    assert client.post("/chat/new/stream", data={"question": "Renewal?"}).status_code == 200

    with db.cursor() as cur:
        cur.execute("INSERT INTO review_queue (kind, payload) VALUES ('other', '{}') RETURNING id")
        row_id = cur.fetchone()[0]
    client.post(f"/review/{row_id}/resolve", follow_redirects=False)

    source = make_source_row(db, "s-signin-call")
    action = build_action("Send the quote", owner_mention="Sam Okafor", locator="00:10:00")
    action_id = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE_MEDICAL)
    client.post(f"/actions/{action_id.action_id}/done", follow_redirects=False)

    with db.cursor() as cur:
        cur.execute("SELECT DISTINCT owner FROM conversations")
        assert cur.fetchall() == [(IDENTITY,)]
        cur.execute("SELECT DISTINCT user_name, kind FROM activity_events ORDER BY kind")
        kinds = cur.fetchall()
        cur.execute("SELECT resolved_by FROM review_queue WHERE id = %s", (row_id,))
        assert cur.fetchone() == (IDENTITY,)
    assert {k for _, k in kinds} >= {"chat_question", "review_decision", "action_confirm"}
    assert {u for u, _ in kinds} == {IDENTITY}


def test_two_people_with_one_name_do_not_share_a_chat_history(
    db, db_url, tmp_path, fake, monkeypatch
) -> None:
    monkeypatch.setattr(chat_routes, "answer_question", lambda *a, **k: _cited_answer(**k))
    with TestClient(_app(db, db_url, tmp_path, fake)) as c:
        second = _signed_in(c, fake, _person("sam.lee@pilot.example"))
        first = _signed_in(c, fake, _person("sam@pilot.example"))
        c.post("/chat/new/stream", data={"question": "Only mine?"})
        with db.cursor() as cur:
            cur.execute("SELECT id FROM conversations")
            [(conversation_id,)] = cur.fetchall()
        assert c.get(f"/c/{conversation_id}", follow_redirects=False).status_code == 200
        _use(c, second)
        assert c.get(f"/c/{conversation_id}", follow_redirects=False).status_code == 404
        del first


def test_a_sign_in_records_the_person_as_an_editor(client, fake, db) -> None:
    _signed_in(client, fake)
    _signed_in(client, fake)
    with db.cursor() as cur:
        cur.execute("SELECT email, name, role, sign_in_count FROM web_users")
        assert cur.fetchall() == [("sam@pilot.example", "Sam Lee", "editor", 2)]


def test_admin_pages_are_for_admins_only(db, db_url, tmp_path, fake) -> None:
    with TestClient(_app(db, db_url, tmp_path, fake)) as c:
        member = _signed_in(c, fake)
        admin = _signed_in(c, fake, _person(ADMIN, first_name="Ada", last_name="Admin"))
        for path in ("/admin/usage", "/admin/people"):
            _use(c, member)
            assert c.get(path, follow_redirects=False).status_code == 403, path
            _use(c, admin)
            assert c.get(path).status_code == 200, path
        _use(c, member)
        refused = c.post(
            "/admin/people/role", data={"email": "sam@pilot.example", "role": "viewer"}
        )
        assert refused.status_code == 403
        _use(c, admin)
        people = c.get("/admin/people").text
        assert "sam@pilot.example" in people and f"{ADMIN} <span" in people


def test_an_admin_switch_to_viewer_blocks_changes_at_once_but_not_reading(
    db, db_url, tmp_path, fake, monkeypatch
) -> None:
    monkeypatch.setattr(chat_routes, "answer_question", lambda *a, **k: _cited_answer(**k))
    with db.cursor() as cur:
        cur.execute("INSERT INTO review_queue (kind, payload) VALUES ('other', '{}') RETURNING id")
        row_id = cur.fetchone()[0]
    with TestClient(_app(db, db_url, tmp_path, fake)) as c:
        sam = _signed_in(c, fake)
        admin = _signed_in(c, fake, _person(ADMIN))
        switched = c.post(
            "/admin/people/role",
            data={"email": "sam@pilot.example", "role": "viewer"},
            follow_redirects=False,
        )
        assert switched.status_code == 303

        _use(c, sam)
        assert c.post(f"/review/{row_id}/resolve", follow_redirects=False).status_code == 403
        assert c.post(f"/review/{row_id}/dismiss", follow_redirects=False).status_code == 403
        assert c.post("/actions/1/done", follow_redirects=False).status_code == 403
        assert c.get("/notes/x/edit", follow_redirects=False).status_code == 403
        assert c.post("/notes/x/edit", data={"after_text": "y"}).status_code == 403
        assert c.get("/atlas").status_code == 200
        assert c.post("/chat/new/stream", data={"question": "Still?"}).status_code == 200

        _use(c, admin)
        c.post("/admin/people/role", data={"email": "sam@pilot.example", "role": "editor"})
        _use(c, sam)
        assert c.post(f"/review/{row_id}/dismiss", follow_redirects=False).status_code == 303
    with db.cursor() as cur:
        cur.execute("SELECT role, role_changed_by FROM web_users WHERE email = 'sam@pilot.example'")
        assert cur.fetchone() == ("editor", f"Sam Lee <{ADMIN}>")


# ---------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------


def test_logout_clears_the_cookie_and_ends_the_workos_session(client, fake) -> None:
    _signed_in(client, fake)
    resp = client.post("/logout", follow_redirects=False)
    assert resp.status_code == 200
    expected = f"{FAKE_AUTHKIT}/logout?session_id=session_sam&amp;return_to={BASE}/signed-out"
    assert f'content="0;url={expected}"' in resp.text
    assert SESSION_COOKIE_NAME not in client.cookies
    assert (
        client.get("/", headers={"Accept": "text/html"}, follow_redirects=False).headers["location"]
        == "/login?next=/"
    )


# ---------------------------------------------------------------------
# The MCP door keeps its own auth
# ---------------------------------------------------------------------

MCP_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "signin-test", "version": "0.0.1"},
    },
}
MCP_ACCEPT = "application/json, text/event-stream"


def test_the_mcp_endpoint_is_not_behind_the_web_sign_in(db, db_url, tmp_path) -> None:
    app = _app(db, db_url, tmp_path, mcp_org_key=MCP_KEY)
    with TestClient(app) as c:
        anonymous = c.post("/mcp", json=MCP_INIT, headers={"Accept": MCP_ACCEPT})
        assert anonymous.status_code == 401
        assert "location" not in anonymous.headers
        assert anonymous.json()["error"] == "unauthorized"
        keyed = c.post(
            "/mcp",
            json=MCP_INIT,
            headers={"Accept": MCP_ACCEPT, "Authorization": f"Bearer {MCP_KEY}"},
        )
        assert keyed.status_code == 200
        assert "serverInfo" in keyed.text


def test_the_oauth_metadata_stays_public_under_workos(db, db_url, tmp_path) -> None:
    app = _app(
        db,
        db_url,
        tmp_path,
        public_base_url="https://portolan-test.example.com",
        mcp_oauth_issuer="https://issuer.example.test",
        mcp_oauth_org_id=ORG,
    )
    with TestClient(app) as c:
        for path in (
            "/.well-known/oauth-protected-resource/mcp",
            "/.well-known/oauth-protected-resource",
        ):
            resp = c.get(path, follow_redirects=False)
            assert resp.status_code == 200, path
            assert resp.json()["authorization_servers"] == ["https://issuer.example.test"]
        challenge = c.post("/mcp", json=MCP_INIT, headers={"Accept": MCP_ACCEPT})
        assert challenge.status_code == 401
        assert "resource_metadata=" in challenge.headers["www-authenticate"]


# ---------------------------------------------------------------------
# web_auth="none" is unchanged
# ---------------------------------------------------------------------


def test_with_web_auth_none_the_shared_key_login_works_as_before(db, db_url, tmp_path) -> None:
    app = _app(db, db_url, tmp_path, web_auth="none", workos_client_id=None, workos_api_key=None)
    assert app.state.workos_provider is None
    with TestClient(app) as c:
        assert c.get("/how-it-works").status_code in (200, 404)  # public, as before
        assert c.get("/search?q=x", follow_redirects=False).status_code == 302
        form = c.get("/login")
        assert form.status_code == 200 and 'name="org_key"' in form.text
        for path in ("/auth/callback?code=c&state=s", "/signed-out"):
            assert c.get(path, follow_redirects=False).status_code == 404, path
        assert c.get("/", follow_redirects=False).status_code == 302
        login = c.post(
            "/login", data={"org_key": ORG_KEY, "editor": "Pat Typed"}, follow_redirects=False
        )
        assert login.status_code == 303
        assert "Pat Typed" in c.get("/").text
        assert c.get("/admin/usage").status_code == 200  # any signed-in reader, as before
        assert c.get("/admin/people", follow_redirects=False).status_code == 404
        record_activity(db, kind="page_view", user_name="x")
        logout = c.post("/logout", follow_redirects=False)
        assert (logout.status_code, logout.headers["location"]) == (303, "/login")


def test_with_web_auth_none_a_workos_cookie_is_not_a_session(db, db_url, tmp_path) -> None:
    app = _app(db, db_url, tmp_path, web_auth="none")
    cookie = URLSafeTimedSerializer(SESSION_SECRET, salt=WORKOS_SESSION_COOKIE_SALT).dumps(
        {"email": "sam@pilot.example", "name": "Sam"}
    )
    with TestClient(app) as c:
        c.cookies.set(SESSION_COOKIE_NAME, cookie)
        assert c.get("/", follow_redirects=False).status_code == 302


def test_web_auth_defaults_to_none(tmp_path) -> None:
    assert Settings(_env_file=None, store_root=tmp_path).web_auth == "none"
