"""`/login`, `/logout`, and under web_auth="workos" the AuthKit callback.

web_auth="none": one shared key (`Settings.org_key`) plus a required display
name (posted as `editor`, 2 to 40 characters) that attributes edits and owns the
reader's chat conversations.

web_auth="workos" (`portolan.web.signin`): `GET /login` sends the browser to
WorkOS AuthKit; `GET /auth/callback` checks the returned state, exchanges the
code, applies the organization and domain policy and sets the session;
`POST /logout` also ends the WorkOS session; `GET /signed-out` is where WorkOS
sends the browser afterwards. `POST /login` sets no session in this mode.

`POST /login` is throttled by `LoginRateLimiter`, an in-process guard on
`request.app.state`, created lazily so each app instance (and each test's
app) gets its own. Its knobs are `Settings` fields (`login_rate_limit_*`,
`trust_forwarded_for`).
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field

import psycopg
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from portolan.config import secret_value
from portolan.web import signin
from portolan.web.auth import (
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    create_session_cookie,
    create_workos_session_cookie,
    get_session,
    verify_org_key,
)
from portolan.web.deps import get_conn
from portolan.web.people import record_sign_in

router = APIRouter()
_logger = logging.getLogger(__name__)

DISPLAY_NAME_MIN_CHARS = 2
DISPLAY_NAME_MAX_CHARS = 40

# Per app instance, not a module global, so failure counts never leak
# between apps sharing a process.
_RATE_LIMITER_STATE_ATTR = "login_rate_limiter"


def _now() -> float:
    """`time.monotonic()`: only elapsed durations are measured, so wall-clock
    steps must not matter. A function so tests can monkeypatch it."""
    return time.monotonic()


@dataclass
class _IPLoginState:
    failure_times: list[float] = field(default_factory=list)
    blocked_until: float = 0.0
    lockout_count: int = 0


class LoginRateLimiter:
    """Per-IP and global brute-force throttle for `POST /login`.

    Per IP: `per_ip_limit` failures inside a rolling `window_seconds` trip
    a lockout starting at `backoff_base_seconds` and doubling on each
    further lockout for that IP, capped at `backoff_max_seconds` (defaults:
    5 per minute, backoff up to 15 minutes).

    Global: `global_limit_per_minute` failures across all IPs in the last
    60 seconds block every caller until they age out, so many IPs with
    few guesses each cannot walk through the per-IP limit.

    State is this process's memory, behind a lock (the sync route runs in
    a threadpool); it resets on restart.
    """

    def __init__(
        self,
        *,
        per_ip_limit: int,
        window_seconds: float,
        backoff_base_seconds: float,
        backoff_max_seconds: float,
        global_limit_per_minute: int,
    ) -> None:
        self._per_ip_limit = max(1, per_ip_limit)
        self._window_seconds = max(1.0, window_seconds)
        self._backoff_base = max(1.0, backoff_base_seconds)
        self._backoff_max = max(self._backoff_base, backoff_max_seconds)
        self._global_limit_per_minute = max(1, global_limit_per_minute)
        self._lock = threading.Lock()
        self._ip_state: dict[str, _IPLoginState] = {}
        self._global_failures: list[float] = []

    def retry_after(self, ip: str) -> float | None:
        """Seconds until `ip` may attempt a login, or `None` when it may try
        now. Call before verifying the key, so a blocked caller gets no
        timing signal about its guess."""
        with self._lock:
            now = _now()
            state = self._ip_state.get(ip)
            if state is not None and state.blocked_until > now:
                return state.blocked_until - now
            self._prune_global(now)
            if len(self._global_failures) >= self._global_limit_per_minute:
                return 60.0
            return None

    def record_failure(self, ip: str) -> None:
        with self._lock:
            now = _now()
            self._prune_global(now)
            self._global_failures.append(now)
            state = self._ip_state.setdefault(ip, _IPLoginState())
            state.failure_times = [t for t in state.failure_times if now - t < self._window_seconds]
            state.failure_times.append(now)
            if len(state.failure_times) >= self._per_ip_limit:
                backoff = min(self._backoff_base * (2**state.lockout_count), self._backoff_max)
                state.blocked_until = now + backoff
                state.lockout_count += 1
                state.failure_times = []

    def record_success(self, ip: str) -> None:
        """A verified org key clears that IP's failure history (the global
        count is kept)."""
        with self._lock:
            self._ip_state.pop(ip, None)

    def _prune_global(self, now: float) -> None:
        cutoff = now - 60.0
        while self._global_failures and self._global_failures[0] < cutoff:
            self._global_failures.pop(0)


def _get_limiter(request: Request, settings) -> LoginRateLimiter:  # noqa: ANN001
    limiter = getattr(request.app.state, _RATE_LIMITER_STATE_ATTR, None)
    if limiter is None:
        limiter = LoginRateLimiter(
            per_ip_limit=settings.login_rate_limit_per_ip,
            window_seconds=settings.login_rate_limit_window_seconds,
            backoff_base_seconds=settings.login_rate_limit_backoff_base_seconds,
            backoff_max_seconds=settings.login_rate_limit_backoff_max_seconds,
            global_limit_per_minute=settings.login_rate_limit_global_per_minute,
        )
        setattr(request.app.state, _RATE_LIMITER_STATE_ATTR, limiter)
    return limiter


def _client_ip(request: Request, *, trust_forwarded_for: bool) -> str:
    """The address to key the rate limiter on. Behind a reverse proxy
    `request.client.host` is the proxy, so a deployment can opt in to
    `X-Forwarded-For` (`trust_forwarded_for`), which reads its leftmost
    entry. That entry is the real client only if the proxy overwrites an
    incoming header; a proxy that appends leaves a client-supplied value
    leftmost, letting a caller pick its own bucket."""
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    if request.client is not None:
        return request.client.host
    return "unknown"


def _format_wait(seconds: float) -> str:
    seconds = max(1, round(seconds))
    if seconds < 60:
        return "less than a minute"
    minutes = max(1, round(seconds / 60))
    return f"about {minutes} minute{'s' if minutes != 1 else ''}"


def _safe_next_path(value: str) -> str:
    """`value` when it is a same-site relative path, else `"/"`, so the
    `?next=` target (set by `app.py`'s 401 redirect) cannot be an absolute
    or `//host` open redirect."""
    if value.startswith("/") and not value.startswith("//") and "\\" not in value:
        return value
    return "/"


@router.get("/login")
def login_form(request: Request, next: str | None = None) -> object:
    templates = request.app.state.templates
    if get_session(request) is not None:
        return RedirectResponse(url="/", status_code=303)
    safe_next = _safe_next_path(next) if next else ""
    if request.app.state.settings.web_auth == "workos":
        return _start_workos_sign_in(request, safe_next or "/")
    return templates.TemplateResponse(
        request, "login.html", {"error": None, "editor": "", "next": safe_next}
    )


def _set_cookie(response, name: str, value: str, *, max_age: int, settings) -> None:  # noqa: ANN001
    # The app's only CSRF defence is `samesite="lax"` (the cookie is withheld
    # from cross-site POSTs); there is no CSRF token.
    response.set_cookie(
        name,
        value,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=max_age,
    )


def _start_workos_sign_in(request: Request, next_path: str) -> RedirectResponse:
    settings = request.app.state.settings
    state = secrets.token_urlsafe(32)
    verifier, challenge = signin.new_pkce_pair()
    url = request.app.state.workos_provider.authorization_url(
        redirect_uri=signin.redirect_uri(settings), state=state, code_challenge=challenge
    )
    response = RedirectResponse(url=url, status_code=303)
    _set_cookie(
        response,
        signin.FLOW_COOKIE_NAME,
        signin.create_flow_cookie(
            state=state,
            code_verifier=verifier,
            next_path=next_path,
            secret=settings.session_secret.get_secret_value(),
        ),
        max_age=signin.FLOW_MAX_AGE_SECONDS,
        settings=settings,
    )
    return response


_REFUSAL_COPY = {
    "organization": "Your account isn't a member of this Portolan's organization. Ask the "
    "person who runs the pilot to invite you.",
    "domain": "Your email address isn't at a domain this Portolan allows.",
    "email_unverified": "Your email address isn't verified yet. Verify it, then sign in again.",
    "state": "That sign-in link expired or was already used. Start again.",
    "provider_error": "The sign-in service turned the request down.",
}


def _refused(request: Request, reason: str, *, logout_url: str | None = None) -> object:
    # The reason code only: never the email, the code, or a token.
    _logger.warning("web sign-in refused: reason=%s", reason)
    message = _REFUSAL_COPY.get(reason, "The sign-in didn't complete. Start again.")
    response = request.app.state.templates.TemplateResponse(
        request,
        "signin_refused.html",
        {"message": message, "reason": reason, "logout_url": logout_url},
        status_code=403,
    )
    response.delete_cookie(signin.FLOW_COOKIE_NAME)
    return response


@router.get(signin.CALLBACK_PATH, include_in_schema=False)
def workos_callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008 - FastAPI dependency
) -> object:
    settings = request.app.state.settings
    if settings.web_auth != "workos":
        raise HTTPException(status_code=404, detail="Not Found")
    secret = settings.session_secret.get_secret_value()
    flow = signin.read_flow_cookie(
        request.cookies.get(signin.FLOW_COOKIE_NAME), state=state, secret=secret
    )
    if flow is None:
        return _refused(request, "state")
    if error or not code:
        return _refused(request, "provider_error")
    code_verifier, next_path = flow
    provider = request.app.state.workos_provider
    try:
        person = provider.authenticate(
            code=code,
            code_verifier=code_verifier,
            organization_id=signin.effective_org_id(settings),
        )
    except signin.SignInError as exc:
        return _refused(request, exc.reason)
    reason = signin.check_access(person, settings)
    if reason is not None:
        logout_url = (
            provider.logout_url(
                session_id=person.session_id, return_to=signin.signed_out_url(settings)
            )
            if person.session_id
            else None
        )
        return _refused(request, reason, logout_url=logout_url)

    email = person.email.strip().lower()
    record_sign_in(conn, email=email, name=person.name, workos_user_id=person.user_id)
    _logger.info("web sign-in ok")
    response = RedirectResponse(url=_safe_next_path(next_path), status_code=303)
    response.delete_cookie(signin.FLOW_COOKIE_NAME)
    _set_cookie(
        response,
        SESSION_COOKIE_NAME,
        create_workos_session_cookie(
            name=person.name, email=email, session_id=person.session_id, secret=secret
        ),
        max_age=SESSION_MAX_AGE_SECONDS,
        settings=settings,
    )
    return response


@router.get(signin.SIGNED_OUT_PATH, include_in_schema=False)
def signed_out(request: Request) -> object:
    if request.app.state.settings.web_auth != "workos":
        raise HTTPException(status_code=404, detail="Not Found")
    return request.app.state.templates.TemplateResponse(request, "signed_out.html", {})


@router.post("/login")
def login_submit(
    request: Request,
    org_key: str = Form(...),
    editor: str = Form(""),
    next: str = Form(""),
) -> object:
    templates = request.app.state.templates
    settings = request.app.state.settings
    if settings.web_auth == "workos":
        # The org key is not a way in under WorkOS sign-in.
        return RedirectResponse(url="/login", status_code=303)
    limiter = _get_limiter(request, settings)
    ip = _client_ip(request, trust_forwarded_for=settings.trust_forwarded_for)
    safe_next = _safe_next_path(next) if next else ""

    wait = limiter.retry_after(ip)
    if wait is not None:
        response = templates.TemplateResponse(
            request,
            "login.html",
            {
                "error": (
                    f"Too many attempts from this address. Wait {_format_wait(wait)} and try again."
                ),
                "editor": editor,
                "next": safe_next,
            },
            status_code=429,
        )
        response.headers["Retry-After"] = str(max(1, round(wait)))
        return response

    if not verify_org_key(org_key, secret_value(settings.org_key)):
        limiter.record_failure(ip)
        # IP only, never the submitted key.
        _logger.warning("login failed: ip=%s", ip)
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "error": "That organization key doesn't match. Check it and try again.",
                "editor": editor,
                "next": safe_next,
            },
            status_code=401,
        )

    limiter.record_success(ip)

    editor_name = editor.strip()
    if not (DISPLAY_NAME_MIN_CHARS <= len(editor_name) <= DISPLAY_NAME_MAX_CHARS):
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "error": (
                    f"Enter your name ({DISPLAY_NAME_MIN_CHARS} to {DISPLAY_NAME_MAX_CHARS} "
                    "characters) -- it's used to attribute your edits and questions, and "
                    "it's whose chat history you'll see."
                ),
                "editor": editor,
                "next": safe_next,
            },
            status_code=400,
        )

    response = RedirectResponse(url=safe_next or "/", status_code=303)
    _set_cookie(
        response,
        SESSION_COOKIE_NAME,
        create_session_cookie(editor_name, secret=settings.session_secret.get_secret_value()),
        max_age=SESSION_MAX_AGE_SECONDS,
        settings=settings,
    )
    return response


@router.post("/logout")
def logout(request: Request) -> object:
    settings = request.app.state.settings
    if settings.web_auth != "workos":
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE_NAME)
        return response
    # Ending only Portolan's cookie would let AuthKit sign the person straight back
    # in, so the browser is also sent to WorkOS's logout URL. A page, not a 303:
    # the CSP's `form-action 'self'` would block a form POST redirected to WorkOS.
    session = get_session(request)
    logout_url = signin.SIGNED_OUT_PATH
    if session is not None and session.session_id:
        logout_url = request.app.state.workos_provider.logout_url(
            session_id=session.session_id, return_to=signin.signed_out_url(settings)
        )
    response = request.app.state.templates.TemplateResponse(
        request, "signing_out.html", {"logout_url": logout_url}
    )
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


__all__ = ["LoginRateLimiter", "router"]
