"""Web sign-in sessions, the role checks, and the query log.

Two sign-in modes (`Settings.web_auth`):

- "none": one shared org key (`Settings.org_key`) plus a typed display name.
- "workos": WorkOS AuthKit (`portolan.web.signin`); the verified name and email
  go in the cookie, and `SignInGateMiddleware` requires a session on every path
  but the public ones.

Either way the session is a signed, expiring cookie keyed by
`Settings.session_secret`, under a salt per mode, so a cookie minted in one mode
never reads as a session in the other. Every chat answer and every original
opened writes a `usage_events` row, never a note body.
"""

from __future__ import annotations

import hmac
from datetime import UTC, datetime
from typing import Any

import psycopg
from fastapi import Depends, HTTPException, Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from psycopg.types.json import Jsonb

from portolan.web.deps import get_conn

__all__ = [
    "ORG_KEY_ENV_VAR",
    "SESSION_COOKIE_NAME",
    "SESSION_MAX_AGE_SECONDS",
    "SESSION_SECRET_ENV_VAR",
    "Session",
    "create_session_cookie",
    "create_workos_session_cookie",
    "get_session",
    "is_admin",
    "log_usage_event",
    "read_session_cookie",
    "require_admin",
    "require_admin_api",
    "require_editor",
    "require_editor_page",
    "require_session",
    "require_session_api",
    "session_role",
    "verify_org_key",
    "workos_identity",
]

SESSION_COOKIE_NAME = "portolan_session"
SESSION_COOKIE_SALT = "portolan-session-v1"
WORKOS_SESSION_COOKIE_SALT = "portolan-session-workos-v1"
# Enforced on every read (`max_age` in `read_session_cookie`), not only at issue.
SESSION_MAX_AGE_SECONDS = 60 * 60 * 8

# Names for error messages; the values live on `Settings.org_key` /
# `Settings.session_secret`.
ORG_KEY_ENV_VAR = "PORTOLAN_ORG_KEY"
SESSION_SECRET_ENV_VAR = "PORTOLAN_SESSION_SECRET"


def verify_org_key(provided: str, expected: str | None) -> bool:
    """Constant-time comparison of a submitted org key against the
    configured one. A falsy `expected` always rejects."""
    if not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _serializer(secret: str, *, mode: str = "none") -> URLSafeTimedSerializer:
    salt = WORKOS_SESSION_COOKIE_SALT if mode == "workos" else SESSION_COOKIE_SALT
    return URLSafeTimedSerializer(secret, salt=salt)


# Longest identity string written to attribution columns (`ai_access_log.changed_by`
# checks 1 to 200 characters, migration 0031).
MAX_IDENTITY_CHARS = 200


class Session:
    """The signed-cookie payload.

    `editor` is the attribution string every route writes (`edit_records.editor`,
    conversation owner, activity `user_name`, review and action decisions). Under
    web_auth="none" it is the typed name; under "workos" it is `Name <email>` (or
    the email alone), so two people with one name never share a chat history.
    `email` and `session_id` are set only for a WorkOS sign-in.
    """

    def __init__(
        self,
        editor: str,
        *,
        name: str | None = None,
        email: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.editor = editor
        self.name = name or editor
        self.email = email
        self.session_id = session_id

    @property
    def display_name(self) -> str:
        return self.name


def workos_identity(name: str | None, email: str) -> str:
    """The attribution string for a WorkOS-verified person."""
    identity = f"{name} <{email}>" if name else email
    return identity[:MAX_IDENTITY_CHARS]


def create_session_cookie(editor: str, *, secret: str) -> str:
    """A web_auth="none" session. `secret` is the plain value of `Settings.session_secret`."""
    return _serializer(secret).dumps({"editor": editor, "issued": datetime.now(UTC).isoformat()})


def create_workos_session_cookie(
    *, name: str | None, email: str, session_id: str | None, secret: str
) -> str:
    return _serializer(secret, mode="workos").dumps(
        {
            "name": name or "",
            "email": email,
            "sid": session_id or "",
            "issued": datetime.now(UTC).isoformat(),
        }
    )


def read_session_cookie(value: str, *, secret: str, mode: str = "none") -> Session | None:
    try:
        data = _serializer(secret, mode=mode).loads(value, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(data, dict):
        return None
    if mode == "workos":
        email = data.get("email")
        if not email or not isinstance(email, str):
            return None
        name = str(data.get("name") or "") or None
        return Session(
            editor=workos_identity(name, email),
            name=name or email,
            email=email,
            session_id=str(data.get("sid") or "") or None,
        )
    editor = data.get("editor")
    if not editor:
        return None
    return Session(editor=str(editor))


def get_session(request: Request) -> Session | None:
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw:
        return None
    settings = request.app.state.settings
    if not settings.session_secret:
        # Only reachable for an app not built by `create_app`.
        return None
    return read_session_cookie(
        raw, secret=settings.session_secret.get_secret_value(), mode=settings.web_auth
    )


def require_session(request: Request) -> Session:
    """Dependency for full-page GET routes: a valid `Session`, or a 302
    to `/login` for a browser that doesn't have one -- there's a page to
    send a person to here."""
    session = get_session(request)
    if session is None:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    return session


def require_session_api(request: Request) -> Session:
    """Dependency for POST/JSON/HTMX-fragment endpoints: a valid
    `Session`, or a 401. `app.py`'s error handler turns that 401 into a
    `/login` redirect for a browser page navigation."""
    session = get_session(request)
    if session is None:
        raise HTTPException(status_code=401, detail="login required")
    return session


def is_admin(session: Session, settings) -> bool:  # noqa: ANN001 - Settings, no import cycle
    """Under web_auth="workos", an email in `Settings.admin_emails`. Under "none"
    there is no verified identity, so every signed-in reader passes, as before."""
    if settings.web_auth != "workos":
        return True
    from portolan.web.people import parse_email_list

    return bool(session.email) and session.email.lower() in parse_email_list(settings.admin_emails)


def require_admin(request: Request) -> Session:
    """Dependency for admin pages: a 302 to `/login` without a session, a 403 for a
    signed-in person who is not an admin."""
    session = require_session(request)
    if not is_admin(session, request.app.state.settings):
        raise HTTPException(status_code=403, detail="admins only")
    return session


def require_admin_api(request: Request) -> Session:
    session = require_session_api(request)
    if not is_admin(session, request.app.state.settings):
        raise HTTPException(status_code=403, detail="admins only")
    return session


def session_role(conn: psycopg.Connection, session: Session, settings) -> str:  # noqa: ANN001
    """'editor' or 'viewer'. Always 'editor' under web_auth="none"."""
    if settings.web_auth != "workos" or not session.email:
        return "editor"
    from portolan.web.people import get_role

    return get_role(conn, session.email)


def require_editor(request: Request, conn: psycopg.Connection = Depends(get_conn)) -> Session:  # noqa: B008
    """Dependency for writes (note edits, review decisions, action confirms): a 401
    without a session, a 403 for a viewer. The role is read on every request, so an
    admin's switch applies at once."""
    session = require_session_api(request)
    if session_role(conn, session, request.app.state.settings) != "editor":
        raise HTTPException(status_code=403, detail="viewers cannot change the graph")
    return session


def require_editor_page(request: Request, conn: psycopg.Connection = Depends(get_conn)) -> Session:  # noqa: B008
    """`require_editor` for a full-page GET (the note edit form)."""
    session = require_session(request)
    if session_role(conn, session, request.app.state.settings) != "editor":
        raise HTTPException(status_code=403, detail="viewers cannot change the graph")
    return session


def log_usage_event(
    conn: psycopg.Connection,
    *,
    tenant: str,
    surface: str,
    model_key: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    cost_estimate: float | None = None,
    request_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """One `usage_events` row: the query and access log (portolan-spec-v2.md
    sec 12). `extra` holds sizes, slugs, ids and the editor name, never a
    note body, claim text, question, or answer.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO usage_events
                (tenant, surface, model_key, input_tokens, output_tokens,
                 cached_tokens, cost_estimate, request_id, extra)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                tenant,
                surface,
                model_key,
                input_tokens,
                output_tokens,
                cached_tokens,
                cost_estimate,
                request_id,
                Jsonb(extra or {}),
            ),
        )
    conn.commit()
