"""WorkOS AuthKit sign-in for the web app (`Settings.web_auth = "workos"`).

The hosted AuthKit flow (https://workos.com/docs/authkit/vanilla/python, read
2026-09-25): `/login` sends the browser to AuthKit's authorize URL with a random
`state` and a PKCE challenge, both kept in a short-lived signed cookie;
`/auth/callback` checks `state`, exchanges the code server-side with the client ID
and API key ("authenticate with code",
https://workos.com/docs/reference/authkit/authentication/code), applies the
access policy below, and sets Portolan's own session cookie. WorkOS's sealed
session is not used: the app keeps its 8-hour signed cookie
(`portolan.web.auth`), and logout sends the browser to WorkOS's logout URL with
the session ID from the access token's `sid` claim
(https://workos.com/docs/authkit/sessions).

Access policy (`check_access`): when an organization is configured, the
organization WorkOS signed the person in to must equal it; when allowed domains
are configured, the person's email must be verified and at one of them. Both
apply when both are set. `validate_config` refuses a configuration with neither.

`SignInGateMiddleware` requires a session on every path except `PUBLIC_PATHS`,
`PUBLIC_PREFIXES` and the MCP endpoints, which keep their own auth.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

import jwt
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from starlette.requests import cookie_parser

from portolan.config import Settings
from portolan.web.auth import SESSION_COOKIE_NAME, read_session_cookie
from portolan.web.people import parse_email_list

__all__ = [
    "CALLBACK_PATH",
    "FLOW_COOKIE_NAME",
    "FLOW_MAX_AGE_SECONDS",
    "PUBLIC_PATHS",
    "PUBLIC_PREFIXES",
    "SIGNED_OUT_PATH",
    "SdkWorkOSProvider",
    "SignInError",
    "SignInGateMiddleware",
    "SignedInPerson",
    "WorkOSProvider",
    "allowed_domains",
    "build_provider",
    "check_access",
    "create_flow_cookie",
    "effective_org_id",
    "new_pkce_pair",
    "read_flow_cookie",
    "redirect_uri",
    "signed_out_url",
    "validate_config",
]

_logger = logging.getLogger(__name__)

FLOW_COOKIE_NAME = "portolan_signin_flow"
_FLOW_COOKIE_SALT = "portolan-signin-flow-v1"
# Long enough to finish a Google or Microsoft sign-in, short enough that a stale
# flow cookie cannot be replayed later.
FLOW_MAX_AGE_SECONDS = 600

CALLBACK_PATH = "/auth/callback"
SIGNED_OUT_PATH = "/signed-out"

# Paths that never need a session. `/mcp` and the OAuth metadata keep their own auth
# (docs/mcp-hosted.md). `/t/o/` and `/t/c/` are the Monday email's
# open pixel and click redirect: email clients fetch them with no cookie, their
# tokens are signed, and the click target is itself gated.
PUBLIC_PATHS = frozenset({"/healthz", "/login", "/logout", CALLBACK_PATH, SIGNED_OUT_PATH, "/mcp"})
PUBLIC_PREFIXES = ("/static/", "/.well-known/oauth-protected-resource", "/t/o/", "/t/c/")


@dataclass(frozen=True)
class SignedInPerson:
    """What Portolan keeps from a WorkOS "authenticate with code" response."""

    user_id: str
    email: str
    email_verified: bool
    first_name: str | None
    last_name: str | None
    organization_id: str | None
    session_id: str | None

    @property
    def name(self) -> str | None:
        parts = [p.strip() for p in (self.first_name, self.last_name) if p and p.strip()]
        return " ".join(parts) or None


class SignInError(Exception):
    """A sign-in that must be refused. `reason` is a short code, safe to log and show."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class WorkOSProvider(Protocol):
    def authorization_url(self, *, redirect_uri: str, state: str, code_challenge: str) -> str: ...

    def authenticate(
        self, *, code: str, code_verifier: str, organization_id: str | None
    ) -> SignedInPerson: ...

    def logout_url(self, *, session_id: str, return_to: str | None) -> str: ...


def _session_id_from_access_token(access_token: str) -> str | None:
    """The `sid` claim, read without checking the signature: the token came straight
    from WorkOS's API over TLS in the server-side code exchange, and `sid` is only
    used to build the logout link."""
    try:
        claims = jwt.decode(access_token, options={"verify_signature": False})
    except jwt.PyJWTError:
        return None
    sid = claims.get("sid")
    return sid if isinstance(sid, str) and sid else None


class SdkWorkOSProvider:
    """`WorkOSProvider` over the official SDK (`workos` on PyPI)."""

    def __init__(self, *, client_id: str, api_key: str) -> None:
        from workos import WorkOSClient

        self._client = WorkOSClient(api_key=api_key, client_id=client_id)

    def authorization_url(self, *, redirect_uri: str, state: str, code_challenge: str) -> str:
        return self._client.user_management.get_authorization_url(
            provider="authkit",
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method="S256",
        )

    def authenticate(
        self, *, code: str, code_verifier: str, organization_id: str | None
    ) -> SignedInPerson:
        from workos import TransportError, WorkOSError
        from workos._errors import OrganizationSelectionRequiredError

        um = self._client.user_management
        try:
            try:
                response = um.authenticate_with_code(code=code, code_verifier=code_verifier)
            except OrganizationSelectionRequiredError as exc:
                # A member of several organizations: sign in to ours. WorkOS refuses
                # if they are not a member of it.
                if not organization_id or not exc.pending_authentication_token:
                    raise SignInError("organization_selection") from None
                response = um.authenticate_with_organization_selection(
                    pending_authentication_token=exc.pending_authentication_token,
                    organization_id=organization_id,
                )
        except SignInError:
            raise
        except (WorkOSError, TransportError) as exc:
            raise SignInError(f"workos_{type(exc).__name__}") from None
        user = response.user
        return SignedInPerson(
            user_id=user.id,
            email=user.email,
            email_verified=bool(user.email_verified),
            first_name=user.first_name,
            last_name=user.last_name,
            organization_id=response.organization_id,
            session_id=_session_id_from_access_token(response.access_token),
        )

    def logout_url(self, *, session_id: str, return_to: str | None) -> str:
        return self._client.user_management.get_logout_url(
            session_id=session_id, return_to=return_to
        )


def effective_org_id(settings: Settings) -> str | None:
    return settings.web_auth_org_id or settings.mcp_oauth_org_id


def allowed_domains(settings: Settings) -> frozenset[str]:
    return frozenset(d.lstrip("@") for d in parse_email_list(settings.web_auth_allowed_domains))


def validate_config(settings: Settings) -> list[str]:
    """Names of what is missing for web_auth="workos"; empty when ready."""
    missing = []
    if not settings.workos_client_id:
        missing.append("PORTOLAN_WORKOS_CLIENT_ID")
    if settings.workos_api_key is None:
        missing.append("PORTOLAN_WORKOS_API_KEY")
    if not settings.public_base_url:
        missing.append("PORTOLAN_PUBLIC_URL")
    if not effective_org_id(settings) and not allowed_domains(settings):
        missing.append(
            "PORTOLAN_WEB_AUTH_ORG_ID (or PORTOLAN_MCP_OAUTH_ORG_ID) or "
            "PORTOLAN_WEB_AUTH_ALLOWED_DOMAINS"
        )
    return missing


def build_provider(settings: Settings) -> WorkOSProvider:
    assert settings.workos_client_id and settings.workos_api_key  # validate_config ran first
    return SdkWorkOSProvider(
        client_id=settings.workos_client_id,
        api_key=settings.workos_api_key.get_secret_value(),
    )


def redirect_uri(settings: Settings) -> str:
    return (settings.public_base_url or "").rstrip("/") + CALLBACK_PATH


def signed_out_url(settings: Settings) -> str:
    return (settings.public_base_url or "").rstrip("/") + SIGNED_OUT_PATH


def check_access(person: SignedInPerson, settings: Settings) -> str | None:
    """None when the person may enter, else a refusal reason code."""
    org_id = effective_org_id(settings)
    if org_id and person.organization_id != org_id:
        return "organization"
    domains = allowed_domains(settings)
    if domains:
        if not person.email_verified:
            return "email_unverified"
        domain = person.email.rpartition("@")[2].lower()
        if domain not in domains:
            return "domain"
    return None


def new_pkce_pair() -> tuple[str, str]:
    """(verifier, S256 challenge), RFC 7636."""
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _flow_serializer(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt=_FLOW_COOKIE_SALT)


def create_flow_cookie(*, state: str, code_verifier: str, next_path: str, secret: str) -> str:
    return _flow_serializer(secret).dumps({"s": state, "v": code_verifier, "n": next_path})


def read_flow_cookie(value: str | None, *, state: str, secret: str) -> tuple[str, str] | None:
    """(code_verifier, next_path) when the cookie is valid, fresh, and its state
    equals the `state` WorkOS sent back; else None."""
    if not value or not state:
        return None
    try:
        data = _flow_serializer(secret).loads(value, max_age=FLOW_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(data, dict):
        return None
    expected = str(data.get("s") or "")
    if not expected or not hmac.compare_digest(expected.encode(), state.encode()):
        return None
    return str(data.get("v") or ""), str(data.get("n") or "/")


def _is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


class SignInGateMiddleware:
    """Pure ASGI: every HTTP request outside the public paths needs a valid WorkOS
    session cookie. A browser page navigation (a GET whose `Accept` lists
    text/html) is sent to `/login?next=<path>`; anything else gets a 401. The
    per-route `require_session` dependencies stay as a second check."""

    def __init__(self, app: Any, *, settings: Settings) -> None:
        self.app = app
        assert settings.session_secret is not None
        self._secret = settings.session_secret.get_secret_value()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or _is_public(scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        if self._has_session(headers.get("cookie", "")):
            await self.app(scope, receive, send)
            return
        # An HTMX swap gets the 401 too, as `app.py`'s error handler does, so a
        # fragment target never receives the login redirect's page.
        is_page = (
            scope.get("method") == "GET"
            and "text/html" in headers.get("accept", "")
            and headers.get("hx-request", "").lower() != "true"
        )
        if is_page:
            location = "/login?next=" + quote(scope.get("path", "/"), safe="/")
            await _send(send, 303, [(b"location", location.encode("latin-1"))], b"")
            return
        await _send(
            send, 401, [(b"content-type", b"application/json")], b'{"detail":"login required"}'
        )

    def _has_session(self, cookie_header: str) -> bool:
        value = cookie_parser(cookie_header).get(SESSION_COOKIE_NAME)
        if not value:
            return False
        return read_session_cookie(value, secret=self._secret, mode="workos") is not None


async def _send(send: Any, status: int, headers: list[tuple[bytes, bytes]], body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [*headers, (b"content-length", str(len(body)).encode("ascii"))],
        }
    )
    await send({"type": "http.response.body", "body": body})
