"""OAuth on the MCP door (portolan/mcp_server/oauth.py, `server.OAuthGateMiddleware`).

A fake authorization server runs in-process: a real HTTP server on a loopback port serving
RFC 8414 metadata and a JWKS for an RSA key generated here. Tokens are minted with that key
and sent to the real `build_http_app` / `create_app` output, so discovery, key fetching,
signature checks and the 401/403 responses are all exercised end to end.

Each rejection test starts from a token that is valid in every other respect and asserts the
logged reason code, so a test cannot pass because the token failed some unrelated check.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx2
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from starlette.testclient import TestClient

from portolan.config import Settings
from portolan.mcp_server.access import AI_ACCESS_CLOSED_CODE, set_ai_access
from portolan.mcp_server.oauth import (
    JWTTokenVerifier,
    OAuthResourceConfig,
    SigningKeys,
    TokenRejected,
    _parse_jwks,
    load_oauth_config,
    metadata_candidates,
)
from portolan.mcp_server.server import TOOL_NAMES, build_http_app, create_server
from portolan.web.app import create_app
from tests.test_mcp_server import _lifespan_started

TENANT = "test-mcp-oauth"
PUBLIC_URL = "https://portolan-test.example.com"
RESOURCE = f"{PUBLIC_URL}/mcp"
METADATA_URL = f"{PUBLIC_URL}/.well-known/oauth-protected-resource/mcp"
ORG = "org_01PORTOLANTEST"
STATIC_KEY = "static-mcp-key-for-oauth-tests-0123456789abcdef"
ACCEPT = "application/json, text/event-stream"
DROP = object()

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "oauth-test", "version": "0.0.1"},
    },
}


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


@pytest.fixture(scope="module")
def rsa_keys() -> dict[str, rsa.RSAPrivateKey]:
    return {
        name: rsa.generate_private_key(public_exponent=65537, key_size=2048)
        for name in ("issuer", "issuer-next", "attacker")
    }


class FakeIssuer:
    """An authorization server's public face: metadata and JWKS over real loopback HTTP."""

    def __init__(self, keys: dict[str, rsa.RSAPrivateKey]) -> None:
        self.private_keys = keys
        self.published: dict[str, rsa.RSAPublicKey] = {"key-1": keys["issuer"].public_key()}
        self.metadata_issuer: str | None = None
        self.requests: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 -- http.server's naming
                fake.requests.append(self.path)
                if self.path == "/.well-known/oauth-authorization-server":
                    body = fake.metadata()
                elif self.path == "/oauth2/jwks":
                    body = fake.jwks()
                else:
                    self.send_response(404)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                raw = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.issuer = f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def metadata(self) -> dict[str, Any]:
        return {
            "issuer": self.metadata_issuer or self.issuer,
            "jwks_uri": f"{self.issuer}/oauth2/jwks",
            "authorization_endpoint": f"{self.issuer}/oauth2/authorize",
            "token_endpoint": f"{self.issuer}/oauth2/token",
            "code_challenge_methods_supported": ["S256"],
        }

    def jwks(self) -> dict[str, Any]:
        keys = []
        for kid, public_key in self.published.items():
            entry = jwt.algorithms.RSAAlgorithm.to_jwk(public_key, as_dict=True)
            keys.append({**entry, "kid": kid, "use": "sig", "alg": "RS256"})
        return {"keys": keys}

    def claims(self, **overrides: Any) -> dict[str, Any]:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": self.issuer,
            "aud": RESOURCE,
            "sub": "user_01ALICE",
            "client_id": "client_claude",
            "org_id": ORG,
            "scope": "openid profile email offline_access",
            "iat": now,
            "exp": now + 300,
            "jti": "token-1",
        }
        for name, value in overrides.items():
            if value is DROP:
                claims.pop(name, None)
            else:
                claims[name] = value
        return claims

    def mint(self, *, key: str = "issuer", kid: str = "key-1", **overrides: Any) -> str:
        return jwt.encode(
            self.claims(**overrides),
            self.private_keys[key],
            algorithm="RS256",
            headers={"kid": kid},
        )


@pytest.fixture()
def issuer(rsa_keys):
    fake = FakeIssuer(rsa_keys)
    yield fake
    fake.close()


def _settings(fake: FakeIssuer, tmp_path, **extra: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": "postgresql://localhost/unused-for-this-test",
        "tenant": TENANT,
        "embed_provider": "fake",
        "store_root": tmp_path,
        "public_base_url": PUBLIC_URL,
        "mcp_oauth_issuer": fake.issuer,
        "mcp_oauth_org_id": ORG,
    }
    values.update(extra)
    return Settings(_env_file=None, **values)


def _config(fake: FakeIssuer, tmp_path, **extra: Any) -> OAuthResourceConfig:
    config = load_oauth_config(_settings(fake, tmp_path, **extra))
    assert config is not None
    return config


@pytest.fixture()
def client(issuer, tmp_path):
    settings = _settings(issuer, tmp_path)
    app = build_http_app(
        create_server(settings), org_key=STATIC_KEY, oauth=load_oauth_config(settings)
    )
    with TestClient(app) as test_client:
        yield test_client


def _post(client: TestClient, authorization: str | None, body: dict | None = None, **headers):
    sent = {"Accept": ACCEPT, **headers}
    if authorization is not None:
        sent["Authorization"] = authorization
    return client.post("/mcp", json=body or INITIALIZE, headers=sent)


def _challenge_params(header: str) -> dict[str, str]:
    assert header.startswith("Bearer ")
    params = {}
    for part in header[len("Bearer ") :].split('", '):
        name, _, value = part.partition("=")
        params[name.strip()] = value.strip().strip('"')
    return params


def _logged_reasons(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage().rsplit("reason=", 1)[1]
        for r in caplog.records
        if r.getMessage().startswith("mcp auth failed:")
    ]


# --- settings --------------------------


def test_oauth_is_off_when_no_issuer_is_set(tmp_path, caplog):
    settings = Settings(_env_file=None, store_root=tmp_path, public_base_url=PUBLIC_URL)
    with caplog.at_level("WARNING"):
        assert load_oauth_config(settings) is None
    assert caplog.records == []


def test_a_blank_issuer_variable_counts_as_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTOLAN_MCP_OAUTH_ISSUER", "   ")
    monkeypatch.setenv("PORTOLAN_MCP_OAUTH_ORG_ID", "")
    settings = Settings(_env_file=None, store_root=tmp_path)
    assert settings.mcp_oauth_issuer is None
    assert settings.mcp_oauth_org_id is None
    assert load_oauth_config(settings) is None


def test_the_resource_defaults_to_the_public_url_plus_mcp(issuer, tmp_path):
    config = _config(issuer, tmp_path, public_base_url=f"{PUBLIC_URL}/")
    assert config.resource == RESOURCE
    assert config.metadata_url == METADATA_URL
    assert config.metadata_paths == (
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-protected-resource",
    )
    assert config.required_scopes == ("openid",)
    assert config.tenant == TENANT


def test_an_explicit_resource_wins_over_the_public_url(issuer, tmp_path):
    config = _config(issuer, tmp_path, mcp_oauth_resource="https://mcp.acme.example/door/mcp")
    assert config.resource == "https://mcp.acme.example/door/mcp"
    assert config.metadata_url == (
        "https://mcp.acme.example/.well-known/oauth-protected-resource/door/mcp"
    )


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"mcp_oauth_org_id": None}, "PORTOLAN_MCP_OAUTH_ORG_ID"),
        ({"public_base_url": None}, "PORTOLAN_PUBLIC_URL"),
        ({"mcp_oauth_issuer": "http://auth.example.com"}, "PORTOLAN_MCP_OAUTH_ISSUER"),
        ({"mcp_oauth_required_scopes": "  "}, "PORTOLAN_MCP_OAUTH_REQUIRED_SCOPES"),
        ({"mcp_oauth_jwks_url": "http://keys.example.com/jwks"}, "PORTOLAN_MCP_OAUTH_JWKS_URL"),
    ],
)
def test_an_incomplete_setup_leaves_oauth_off_and_says_why(issuer, tmp_path, caplog, change, named):
    with caplog.at_level("WARNING"):
        assert load_oauth_config(_settings(issuer, tmp_path, **change)) is None
    assert named in caplog.text
    assert "stays off" in caplog.text


def test_metadata_candidates_follow_the_mcp_discovery_order():
    assert metadata_candidates("https://auth.example.com") == [
        "https://auth.example.com/.well-known/oauth-authorization-server",
        "https://auth.example.com/.well-known/openid-configuration",
    ]
    assert metadata_candidates("https://auth.example.com/tenant1") == [
        "https://auth.example.com/.well-known/oauth-authorization-server/tenant1",
        "https://auth.example.com/.well-known/openid-configuration/tenant1",
        "https://auth.example.com/tenant1/.well-known/openid-configuration",
    ]


# --- protected-resource metadata --------------------------


@pytest.mark.parametrize(
    "path", ["/.well-known/oauth-protected-resource/mcp", "/.well-known/oauth-protected-resource"]
)
def test_metadata_is_public_and_names_the_issuer(client, issuer, path):
    resp = client.get(path)

    assert resp.status_code == 200
    assert resp.json() == {
        "resource": RESOURCE,
        "authorization_servers": [issuer.issuer],
        "scopes_supported": ["openid"],
        "bearer_methods_supported": ["header"],
        "resource_name": "Portolan",
    }
    assert resp.headers["access-control-allow-origin"] == "*"
    assert client.options(path).status_code == 204
    assert client.post(path).status_code == 405


# --- the challenge --------------------------


def test_no_credential_is_a_401_that_points_at_the_metadata(client, caplog):
    with caplog.at_level("WARNING"):
        resp = _post(client, None)

    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == (
        f'Bearer resource_metadata="{METADATA_URL}", scope="openid"'
    )
    assert _logged_reasons(caplog) == ["missing"]


def _tampered(fake: FakeIssuer) -> str:
    header, _, signature = fake.mint().split(".")
    payload = _b64url(json.dumps(fake.claims(sub="user_01MALLORY")).encode())
    return f"{header}.{payload}.{signature}"


def _unsigned(fake: FakeIssuer) -> str:
    header = _b64url(json.dumps({"alg": "none", "kid": "key-1"}).encode())
    return f"{header}.{_b64url(json.dumps(fake.claims()).encode())}."


def _hmac_signed(fake: FakeIssuer) -> str:
    return jwt.encode(
        fake.claims(),
        "a-guessable-shared-secret-value!",
        algorithm="HS256",
        headers={"kid": "key-1"},
    )


BAD_TOKENS = {
    "wrong audience": (lambda f: f.mint(aud="https://other.example.com/mcp"), "audience"),
    "audience is the bare host": (lambda f: f.mint(aud=PUBLIC_URL), "audience"),
    "expired": (lambda f: f.mint(exp=int(time.time()) - 3600), "expired"),
    "not yet valid": (lambda f: f.mint(nbf=int(time.time()) + 3600), "not_yet_valid"),
    "wrong issuer": (lambda f: f.mint(iss="https://evil.example.com"), "issuer"),
    "tampered payload": (_tampered, "bad_signature"),
    "signed by another key under the same kid": (lambda f: f.mint(key="attacker"), "bad_signature"),
    "signed by an unknown kid": (lambda f: f.mint(key="attacker", kid="rogue"), "unknown_key"),
    "another organization": (lambda f: f.mint(org_id="org_01SOMEONEELSE"), "org"),
    "no organization": (lambda f: f.mint(org_id=DROP), "org"),
    "no expiry": (lambda f: f.mint(exp=DROP), "missing_claim"),
    "alg none": (_unsigned, "algorithm"),
    "HS256 with a shared secret": (_hmac_signed, "algorithm"),
    "not a JWT": (lambda f: "not-a-jwt", "malformed"),
    "a wrong static key": (lambda f: "wrong-static-key-0123456789abcdef0123", "malformed"),
}


@pytest.mark.parametrize("case", sorted(BAD_TOKENS))
def test_a_bad_token_is_a_401_invalid_token_for_the_right_reason(client, issuer, caplog, case):
    make, reason = BAD_TOKENS[case]
    token = make(issuer)

    with caplog.at_level("WARNING"):
        resp = _post(client, f"Bearer {token}")

    assert resp.status_code == 401
    params = _challenge_params(resp.headers["www-authenticate"])
    assert params["error"] == "invalid_token"
    assert params["resource_metadata"] == METADATA_URL
    assert params["scope"] == "openid"
    assert resp.json()["error"] == "invalid_token"
    assert _logged_reasons(caplog) == [reason]
    assert token not in caplog.text


def test_a_token_without_the_required_scope_is_a_403_naming_the_scope(client, issuer, caplog):
    with caplog.at_level("WARNING"):
        resp = _post(client, f"Bearer {issuer.mint(scope='profile email')}")

    assert resp.status_code == 403
    params = _challenge_params(resp.headers["www-authenticate"])
    assert params["error"] == "insufficient_scope"
    assert params["scope"] == "openid"
    assert params["resource_metadata"] == METADATA_URL
    assert _logged_reasons(caplog) == ["insufficient_scope"]


def test_two_authorization_headers_are_a_400_even_if_one_is_valid(client, issuer):
    resp = client.post(
        "/mcp",
        json=INITIALIZE,
        headers=[
            ("Accept", ACCEPT),
            ("Authorization", f"Bearer {issuer.mint()}"),
            ("Authorization", f"Bearer {STATIC_KEY}"),
        ],
    )
    assert resp.status_code == 400
    assert _challenge_params(resp.headers["www-authenticate"])["error"] == "invalid_request"


def test_a_non_bearer_scheme_is_a_plain_challenge(client):
    resp = _post(client, "Basic dXNlcjpwYXNz")
    assert resp.status_code == 401
    assert "error=" not in resp.headers["www-authenticate"]


# --- what gets through --------------------------


def _initialized(resp) -> str:
    assert resp.status_code == 200
    message = next(
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    )
    assert message["result"]["serverInfo"]["name"] == "portolan"
    return resp.headers["mcp-session-id"]


def test_a_valid_token_opens_an_mcp_session(client, issuer, caplog):
    with caplog.at_level("WARNING"):
        resp = _post(client, f"Bearer {issuer.mint()}")
    assert _initialized(resp)
    assert _logged_reasons(caplog) == []


def test_a_token_whose_audience_list_includes_this_server_is_accepted(client, issuer):
    resp = _post(client, f"Bearer {issuer.mint(aud=['https://other.example.com', RESOURCE + '/'])}")
    assert _initialized(resp)


def test_the_static_key_still_works_with_oauth_on(client, issuer):
    assert _initialized(_post(client, f"Bearer {STATIC_KEY}"))
    assert issuer.requests == []


def test_oauth_only_without_a_static_key(issuer, tmp_path):
    settings = _settings(issuer, tmp_path)
    app = build_http_app(create_server(settings), org_key=None, oauth=load_oauth_config(settings))
    with TestClient(app) as test_client:
        assert _initialized(_post(test_client, f"Bearer {issuer.mint()}"))
        assert _post(test_client, f"Bearer {STATIC_KEY}").status_code == 401


def test_a_session_is_bound_to_the_person_who_opened_it(client, issuer):
    alice = f"Bearer {issuer.mint()}"
    session_id = _initialized(_post(client, alice))
    ping = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    session = {"mcp-session-id": session_id, "mcp-protocol-version": "2025-06-18"}

    bob = f"Bearer {issuer.mint(sub='user_01BOB')}"
    assert _post(client, bob, ping, **session).status_code == 404
    assert _post(client, f"Bearer {STATIC_KEY}", ping, **session).status_code == 404
    assert _post(client, alice, ping, **session).status_code == 200


# --- signing keys and the SDK verifier interface --------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_keys_are_fetched_once_and_reused(issuer, tmp_path):
    verifier = JWTTokenVerifier(_config(issuer, tmp_path))
    verifier.verify(issuer.mint())
    verifier.verify(issuer.mint(sub="user_01BOB"))
    assert issuer.requests == ["/.well-known/oauth-authorization-server", "/oauth2/jwks"]


def test_a_rotated_key_is_picked_up_but_unknown_kids_cannot_force_refetches(issuer, tmp_path):
    config = _config(issuer, tmp_path)
    clock = FakeClock()
    verifier = JWTTokenVerifier(config, keys=SigningKeys(config, clock=clock))
    verifier.verify(issuer.mint())

    issuer.published["key-2"] = issuer.private_keys["issuer-next"].public_key()
    rotated = issuer.mint(key="issuer-next", kid="key-2")
    clock.now += 1
    with pytest.raises(TokenRejected) as excinfo:
        verifier.verify(rotated)
    assert excinfo.value.reason == "unknown_key"
    assert issuer.requests.count("/oauth2/jwks") == 1

    clock.now += 60
    assert verifier.verify(rotated).subject == "user_01ALICE"
    assert issuer.requests.count("/oauth2/jwks") == 2


def test_metadata_naming_a_different_issuer_is_refused(issuer, tmp_path, caplog):
    issuer.metadata_issuer = issuer.issuer + "/"
    verifier = JWTTokenVerifier(_config(issuer, tmp_path))

    with caplog.at_level("WARNING"), pytest.raises(TokenRejected) as excinfo:
        verifier.verify(issuer.mint())

    assert excinfo.value.reason == "jwks_unavailable"
    assert "copy the `issuer` value exactly" in caplog.text
    assert "/oauth2/jwks" not in issuer.requests


def test_an_explicit_jwks_url_skips_discovery(issuer, tmp_path):
    config = _config(issuer, tmp_path, mcp_oauth_jwks_url=f"{issuer.issuer}/oauth2/jwks")
    JWTTokenVerifier(config).verify(issuer.mint())
    assert issuer.requests == ["/oauth2/jwks"]


def test_symmetric_and_encryption_keys_in_a_jwks_are_never_used(issuer):
    document = issuer.jwks()
    document["keys"] += [
        {"kty": "oct", "kid": "shared", "k": _b64url(b"a-guessable-shared-secret-value!")},
        {**document["keys"][0], "kid": "enc", "use": "enc"},
    ]
    assert set(_parse_jwks(document)) == {"key-1"}


def test_verify_token_is_the_sdk_token_verifier_interface(issuer, tmp_path):
    verifier = JWTTokenVerifier(_config(issuer, tmp_path))

    access = asyncio.run(verifier.verify_token(issuer.mint()))
    rejected = asyncio.run(verifier.verify_token(issuer.mint(aud="https://other.example.com")))

    assert access is not None
    assert access.subject == "user_01ALICE"
    assert access.client_id == "client_claude"
    assert access.resource == RESOURCE
    assert "openid" in access.scopes
    assert access.claims == {"iss": issuer.issuer, "org_id": ORG, "tenant": TENANT}
    assert rejected is None


# --- inside the web app --------------------------


def _web_settings(fake: FakeIssuer, db_url: str, tmp_path, **extra: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": db_url,
        "tenant": TENANT,
        "store_root": tmp_path,
        "llm_provider": "fake",
        "embed_provider": "fake",
        "org_key": "unused-web-org-key-for-mcp-oauth-tests",
        "session_secret": "unused-session-secret-for-mcp-oauth-tests",
        "cookie_secure": False,
        "public_base_url": PUBLIC_URL,
        "mcp_oauth_issuer": fake.issuer,
        "mcp_oauth_org_id": ORG,
    }
    values.update(extra)
    return Settings(_env_file=None, **values)


def test_the_web_app_serves_metadata_without_a_login_and_challenges_on_mcp(
    db, db_url, issuer, tmp_path
):
    del db
    app = create_app(_web_settings(issuer, db_url, tmp_path))

    with TestClient(app) as test_client:
        for path in (
            "/.well-known/oauth-protected-resource/mcp",
            "/.well-known/oauth-protected-resource",
        ):
            resp = test_client.get(path, follow_redirects=False)
            assert resp.status_code == 200
            assert resp.json()["authorization_servers"] == [issuer.issuer]

        mcp = test_client.post("/mcp", json=INITIALIZE, headers={"Accept": ACCEPT})
        assert mcp.status_code == 401
        assert _challenge_params(mcp.headers["www-authenticate"])["resource_metadata"] == (
            METADATA_URL
        )

        home = test_client.get("/", follow_redirects=False)
        assert home.status_code == 302


def test_without_oauth_settings_the_web_app_is_exactly_as_before(db, db_url, issuer, tmp_path):
    del db
    app = create_app(
        _web_settings(
            issuer,
            db_url,
            tmp_path,
            mcp_oauth_issuer=None,
            mcp_org_key="hosted-mcp-test-key-0123456789abcdef",
        )
    )

    with TestClient(app) as test_client:
        assert test_client.get("/.well-known/oauth-protected-resource/mcp").status_code == 404
        assert test_client.get("/.well-known/oauth-protected-resource").status_code == 404
        mcp = test_client.post("/mcp", json=INITIALIZE, headers={"Accept": ACCEPT})
        assert mcp.status_code == 401
        assert "www-authenticate" not in mcp.headers
        assert (
            mcp.content
            == b'{"error":"unauthorized","detail":"missing or invalid Authorization header"}'
        )
        rejected = _post(test_client, f"Bearer {issuer.mint()}")
        assert rejected.status_code == 401

    assert issuer.requests == []


def test_an_oauth_client_round_trip_and_the_ai_access_switch_still_applies(
    db, db_url, issuer, tmp_path
):
    set_ai_access(db, TENANT, enabled=True, changed_by="test", reason="oauth round trip")
    app = create_app(_web_settings(issuer, db_url, tmp_path))
    token = issuer.mint()

    async def _run():
        async with (
            _lifespan_started(app),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                base_url="http://testserver",
                headers={"Authorization": f"Bearer {token}"},
            ) as http_client,
            streamable_http_client("http://testserver/mcp", http_client=http_client) as (
                read,
                write,
            ),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            questions = await session.call_tool("list_standing_questions", {})
            set_ai_access(db, TENANT, enabled=False, changed_by="test", reason="switch off")
            try:
                await session.list_tools()
            except MCPError as exc:
                closed = exc
            else:
                closed = None
            return listed, questions, closed

    listed, questions, closed = asyncio.run(_run())

    assert sorted(t.name for t in listed.tools) == sorted(TOOL_NAMES)
    assert questions.is_error is False
    assert closed is not None
    assert closed.code == AI_ACCESS_CLOSED_CODE
