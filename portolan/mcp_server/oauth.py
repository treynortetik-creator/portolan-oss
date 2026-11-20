"""The OAuth resource-server side of the MCP door, for claude.ai and ChatGPT connectors.

Follows the MCP authorization spec, revision 2026-07-28
(https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization): Portolan
never issues tokens. A hosted identity provider (WorkOS AuthKit by default) is the
authorization server; Portolan publishes protected-resource metadata naming it (RFC 9728),
answers an unauthenticated `/mcp` request with a 401 challenge pointing there, and checks
each bearer JWT's signature, issuer, audience, lifetime, organization and scope.

Off unless `PORTOLAN_MCP_OAUTH_ISSUER` is set (`load_oauth_config`). The static
`PORTOLAN_MCP_ORG_KEY` path is untouched either way (`server.OAuthGateMiddleware`).
Operator steps: docs/mcp-hosted.md.
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import anyio.to_thread
import jwt
from mcp.server.auth.provider import AccessToken
from mcp.shared.auth import ProtectedResourceMetadata

from portolan.config import Settings

_logger = logging.getLogger(__name__)

MCP_AUTH_SPEC_REVISION = "2026-07-28"
PROTECTED_RESOURCE_WELL_KNOWN = "/.well-known/oauth-protected-resource"

# Asymmetric only: an HMAC algorithm would let anyone holding the (public) JWKS forge tokens.
ALLOWED_ALGORITHMS = frozenset(
    {"RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512", "EdDSA"}
)
CLOCK_LEEWAY_SECONDS = 30
JWKS_TTL_SECONDS = 3600
# At most one key fetch per interval, so tokens naming unknown key ids cannot make this
# server hammer the identity provider.
JWKS_MIN_REFETCH_SECONDS = 30
HTTP_TIMEOUT_SECONDS = 5
MAX_DOCUMENT_BYTES = 1_000_000

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class TokenRejected(Exception):
    """A bearer token failed one check. `reason` is a short code for the log, never token
    content: malformed, algorithm, unknown_key, jwks_unavailable, bad_signature, expired,
    not_yet_valid, issuer, missing_claim, audience, org."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class OAuthResourceConfig:
    issuer: str
    resource: str
    required_scopes: tuple[str, ...]
    org_claim: str
    org_id: str
    tenant: str
    jwks_url: str | None = None

    @property
    def metadata_url(self) -> str:
        """RFC 9728 sec 3.1: the well-known suffix inserted before the resource's path."""
        parts = urlsplit(self.resource)
        path = parts.path.rstrip("/")
        return f"{parts.scheme}://{parts.netloc}{PROTECTED_RESOURCE_WELL_KNOWN}{path}"

    @property
    def metadata_paths(self) -> tuple[str, ...]:
        """The path-specific document and the root one; MCP clients try them in that order."""
        specific = urlsplit(self.metadata_url).path
        if specific == PROTECTED_RESOURCE_WELL_KNOWN:
            return (specific,)
        return (specific, PROTECTED_RESOURCE_WELL_KNOWN)


def _is_allowed_url(url: str) -> bool:
    """https, or http on a loopback host (local development and tests); no query or fragment."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.query or parts.fragment or not parts.hostname:
        return False
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS


def load_oauth_config(settings: Settings) -> OAuthResourceConfig | None:
    """The OAuth settings, or None when OAuth is off. An incomplete or invalid setup also
    returns None, with one warning naming what is missing: OAuth then stays off (closed),
    and the static key keeps working, rather than the web app failing to boot."""
    issuer = settings.mcp_oauth_issuer
    if issuer is None:
        return None

    problems: list[str] = []
    if not _is_allowed_url(issuer):
        problems.append("PORTOLAN_MCP_OAUTH_ISSUER must be an https:// URL with no query")

    resource = settings.mcp_oauth_resource
    if resource is None and settings.public_base_url:
        resource = settings.public_base_url.strip().rstrip("/") + "/mcp"
    if resource is None:
        problems.append("set PORTOLAN_MCP_OAUTH_RESOURCE, or PORTOLAN_PUBLIC_URL (+ /mcp)")
    elif not _is_allowed_url(resource):
        problems.append("the resource URL must be an https:// URL with no query")

    if settings.mcp_oauth_org_id is None:
        problems.append("PORTOLAN_MCP_OAUTH_ORG_ID is not set")
    org_claim = settings.mcp_oauth_org_claim.strip()
    if not org_claim:
        problems.append("PORTOLAN_MCP_OAUTH_ORG_CLAIM is blank")

    scopes = tuple(settings.mcp_oauth_required_scopes.split())
    if not scopes:
        problems.append("PORTOLAN_MCP_OAUTH_REQUIRED_SCOPES must name at least one scope")

    jwks_url = settings.mcp_oauth_jwks_url
    if jwks_url is not None and not _is_allowed_url(jwks_url):
        problems.append("PORTOLAN_MCP_OAUTH_JWKS_URL must be an https:// URL with no query")

    if problems:
        _logger.warning(
            "MCP OAuth is configured but incomplete, so it stays off (the static MCP key, "
            "if set, still works): %s",
            "; ".join(problems),
        )
        return None

    assert resource is not None and settings.mcp_oauth_org_id is not None
    return OAuthResourceConfig(
        issuer=issuer,
        resource=resource,
        required_scopes=scopes,
        org_claim=org_claim,
        org_id=settings.mcp_oauth_org_id,
        tenant=settings.tenant,
        jwks_url=jwks_url,
    )


# --- protected-resource metadata (RFC 9728) --------------------------


def protected_resource_metadata(config: OAuthResourceConfig) -> dict[str, Any]:
    # Strings, not AnyHttpUrl objects: the model's own `url_preserve_empty_path` then applies,
    # so https://x.authkit.app is not given a trailing slash, which would make it a different
    # issuer to a client comparing it with the identity provider's own metadata.
    document = ProtectedResourceMetadata.model_validate(
        {
            "resource": config.resource,
            "authorization_servers": [config.issuer],
            "scopes_supported": list(config.required_scopes),
            "bearer_methods_supported": ["header"],
            "resource_name": "Portolan",
        }
    )
    return document.model_dump(mode="json", exclude_none=True)


class ProtectedResourceMetadataApp:
    """Serves the metadata document. Public by design: a client reads it before it has
    any credential. CORS-open so a browser-based MCP client can read it too."""

    def __init__(self, config: OAuthResourceConfig) -> None:
        self._body = json.dumps(protected_resource_metadata(config)).encode("utf-8")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        method = scope.get("method", "GET")
        cors = [
            (b"access-control-allow-origin", b"*"),
            (b"access-control-allow-methods", b"GET, OPTIONS"),
            (b"access-control-allow-headers", b"mcp-protocol-version"),
        ]
        if method == "OPTIONS":
            await send({"type": "http.response.start", "status": 204, "headers": cors})
            await send({"type": "http.response.body", "body": b""})
            return
        if method not in ("GET", "HEAD"):
            await send(
                {
                    "type": "http.response.start",
                    "status": 405,
                    "headers": [(b"allow", b"GET, HEAD, OPTIONS"), (b"content-length", b"0")],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(self._body)).encode("ascii")),
            (b"cache-control", b"public, max-age=3600"),
            *cors,
        ]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": b"" if method == "HEAD" else self._body})


# --- the issuer's signing keys --------------------------


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirects)

# URLError and timeouts are OSErrors; a truncated body (IncompleteRead) is not.
_FETCH_ERRORS = (OSError, ValueError, http.client.HTTPException)


def fetch_json(url: str) -> dict[str, Any]:
    """GET a JSON object. No redirects (a 3xx raises), a timeout and a size cap."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "portolan-mcp-oauth"}
    )
    with _OPENER.open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        raw = response.read(MAX_DOCUMENT_BYTES + 1)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"{url} returned more than {MAX_DOCUMENT_BYTES} bytes")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return document


def metadata_candidates(issuer: str) -> list[str]:
    """Authorization-server metadata URLs, in the order the MCP spec has clients try them
    (Authorization Server Discovery, revision 2026-07-28)."""
    parts = urlsplit(issuer)
    base = f"{parts.scheme}://{parts.netloc}"
    path = parts.path.rstrip("/")
    if path:
        return [
            f"{base}/.well-known/oauth-authorization-server{path}",
            f"{base}/.well-known/openid-configuration{path}",
            f"{base}{path}/.well-known/openid-configuration",
        ]
    return [
        f"{base}/.well-known/oauth-authorization-server",
        f"{base}/.well-known/openid-configuration",
    ]


def _parse_jwks(document: dict[str, Any]) -> dict[str | None, jwt.PyJWK]:
    entries = document.get("keys")
    if not isinstance(entries, list):
        raise ValueError("the JWKS document has no `keys` list")
    keys: dict[str | None, jwt.PyJWK] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("kty") == "oct":
            continue
        if entry.get("use") not in (None, "sig"):
            continue
        try:
            key = jwt.PyJWK(entry)
        except jwt.PyJWTError:
            continue
        if key.algorithm_name in ALLOWED_ALGORITHMS:
            kid = entry.get("kid")
            keys[kid if isinstance(kid, str) else None] = key
    return keys


class SigningKeys:
    """The issuer's public keys, fetched on first use and cached: refreshed after
    `JWKS_TTL_SECONDS`, or early when a token names a key id not yet seen (key rotation),
    never more than once per `JWKS_MIN_REFETCH_SECONDS`. A failed refresh keeps the keys
    already held."""

    def __init__(
        self,
        config: OAuthResourceConfig,
        *,
        fetch: Callable[[str], dict[str, Any]] = fetch_json,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._fetch = fetch
        self._clock = clock
        self._lock = threading.Lock()
        self._jwks_url = config.jwks_url
        self._keys: dict[str | None, jwt.PyJWK] = {}
        self._fetched_at: float | None = None
        self._last_attempt: float | None = None

    @property
    def loaded(self) -> bool:
        return bool(self._keys)

    def get(self, kid: str | None) -> jwt.PyJWK | None:
        with self._lock:
            now = self._clock()
            stale = self._fetched_at is None or now - self._fetched_at >= JWKS_TTL_SECONDS
            may_fetch = (
                self._last_attempt is None or now - self._last_attempt >= JWKS_MIN_REFETCH_SECONDS
            )
            if (stale or self._lookup(kid) is None) and may_fetch:
                self._refresh(now)
            return self._lookup(kid)

    def _lookup(self, kid: str | None) -> jwt.PyJWK | None:
        if kid is not None:
            return self._keys.get(kid)
        return next(iter(self._keys.values())) if len(self._keys) == 1 else None

    def _refresh(self, now: float) -> None:
        self._last_attempt = now
        try:
            if self._jwks_url is None:
                self._jwks_url = self._discover_jwks_url()
            keys = _parse_jwks(self._fetch(self._jwks_url))
        except _FETCH_ERRORS as exc:
            _logger.warning(
                "mcp oauth: could not load the issuer's signing keys (%s: %s)",
                type(exc).__name__,
                str(exc)[:300],
            )
            return
        if not keys:
            _logger.warning("mcp oauth: the issuer's JWKS holds no usable signing key")
            return
        self._keys = keys
        self._fetched_at = now

    def _discover_jwks_url(self) -> str:
        issuer = self._config.issuer
        tried: list[str] = []
        for url in metadata_candidates(issuer):
            try:
                document = self._fetch(url)
            except _FETCH_ERRORS:
                tried.append(url)
                continue
            # RFC 8414 sec 3.3: metadata whose issuer differs from the one asked for must not
            # be used. Almost always a typo in the setting, so say so plainly.
            if document.get("issuer") != issuer:
                raise ValueError(
                    f"the metadata at {url} names issuer {document.get('issuer')!r}, not "
                    f"PORTOLAN_MCP_OAUTH_ISSUER {issuer!r}; copy the `issuer` value exactly"
                )
            jwks_uri = document.get("jwks_uri")
            if not isinstance(jwks_uri, str) or not _is_allowed_url(jwks_uri):
                raise ValueError(f"the metadata at {url} has no usable https jwks_uri")
            return jwks_uri
        raise ValueError(f"no authorization server metadata found at {', '.join(tried)}")


# --- bearer token verification --------------------------


def canonical_url(url: str) -> str:
    """For comparing resource URLs: lowercase scheme and host, no trailing slash (MCP spec,
    "Canonical Server URI")."""
    parts = urlsplit(url.strip())
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, "")
    )


def granted_scopes(claims: dict[str, Any]) -> list[str]:
    """`scope` is a space-delimited string (RFC 9068, WorkOS); some issuers use `scp`."""
    raw = claims.get("scope", claims.get("scp"))
    if isinstance(raw, str):
        return raw.split()
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str)]
    return []


class JWTTokenVerifier:
    """Implements the `mcp` SDK's `TokenVerifier` protocol (`mcp.server.auth.provider`).

    `verify` raises `TokenRejected` with the reason; `verify_token` is the SDK's form
    (None on any failure). Scope is not checked here: a token short of scope is a 403,
    not a 401 (`server.OAuthGateMiddleware`)."""

    def __init__(self, config: OAuthResourceConfig, *, keys: SigningKeys | None = None) -> None:
        self._config = config
        self._keys = keys if keys is not None else SigningKeys(config)
        self._resource = canonical_url(config.resource)

    def verify(self, token: str) -> AccessToken:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise TokenRejected("malformed") from None
        alg = header.get("alg")
        if alg not in ALLOWED_ALGORITHMS:
            raise TokenRejected("algorithm")
        kid = header.get("kid")
        key = self._keys.get(kid if isinstance(kid, str) else None)
        if key is None:
            raise TokenRejected("unknown_key" if self._keys.loaded else "jwks_unavailable")

        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=[alg],
                issuer=self._config.issuer,
                leeway=CLOCK_LEEWAY_SECONDS,
                options={"require": ["exp", "iss", "aud", "sub"], "verify_aud": False},
            )
        except jwt.ExpiredSignatureError:
            raise TokenRejected("expired") from None
        except jwt.ImmatureSignatureError:
            raise TokenRejected("not_yet_valid") from None
        except jwt.InvalidIssuerError:
            raise TokenRejected("issuer") from None
        except jwt.InvalidAlgorithmError:
            raise TokenRejected("algorithm") from None
        except jwt.InvalidSignatureError:
            raise TokenRejected("bad_signature") from None
        except jwt.MissingRequiredClaimError:
            raise TokenRejected("missing_claim") from None
        except jwt.PyJWTError:
            raise TokenRejected("malformed") from None

        # MCP spec, Token Handling: the token must name this server as its audience.
        audiences = claims["aud"] if isinstance(claims["aud"], list) else [claims["aud"]]
        if not any(isinstance(a, str) and canonical_url(a) == self._resource for a in audiences):
            raise TokenRejected("audience")

        org = claims.get(self._config.org_claim)
        if not isinstance(org, str) or org != self._config.org_id:
            raise TokenRejected("org")

        subject = claims["sub"]
        client_id = claims.get("client_id") or claims.get("azp") or subject
        return AccessToken(
            token=token,
            client_id=str(client_id),
            scopes=granted_scopes(claims),
            expires_at=int(claims["exp"]),
            resource=self._config.resource,
            subject=str(subject),
            claims={
                "iss": claims["iss"],
                self._config.org_claim: org,
                "tenant": self._config.tenant,
            },
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            return await anyio.to_thread.run_sync(self.verify, token)
        except TokenRejected:
            return None
