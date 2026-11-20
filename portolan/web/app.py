"""`create_app(settings) -> FastAPI` (build-contracts.md "Web and MCP").

At startup (`lifespan`) runs migrations, opens a connection pool, the LLM
client, embedder and store, and stashes them on `app.state`; routes get
them through `portolan.web.deps`, which tests override.

`create_app` raises before building anything when `settings.session_secret`
is unset, or under web_auth="none" when `settings.org_key` is unset, or under
web_auth="workos" when the WorkOS settings are incomplete
(`portolan.web.signin.validate_config`): without them cookies are forgeable or
there is no way to tell who may open the graph. Under "workos",
`SignInGateMiddleware` requires a session on every non-public path.

When `settings.mcp_org_key` or OAuth (`portolan.mcp_server.oauth`) is
configured, the MCP streamable-HTTP transport is served at `/mcp`
(`_register_mcp_route`, `_MCPGateway`); otherwise `/mcp` 404s like any
undefined path.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from psycopg_pool import ConnectionPool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import Route as _ASGIRoute

from portolan import db
from portolan.activity import purge_expired
from portolan.config import Settings, get_settings, secret_value
from portolan.context import seed_company_context
from portolan.llm.factory import get_llm
from portolan.mcp_server.oauth import (
    OAuthResourceConfig,
    ProtectedResourceMetadataApp,
    load_oauth_config,
)
from portolan.mcp_server.server import build_http_app as build_mcp_http_app
from portolan.mcp_server.server import create_server as create_mcp_server
from portolan.search.embed import get_embedder
from portolan.store import LocalStore, assert_store_root_ready
from portolan.web import signin
from portolan.web.auth import ORG_KEY_ENV_VAR, SESSION_SECRET_ENV_VAR, Session, require_session
from portolan.web.routes import (
    actions,
    atlas,
    chat,
    coverage,
    graph,
    health,
    how_it_works,
    login,
    notes,
    originals,
    people,
    reports,
    review,
    search,
    sources,
    usage,
)
from portolan.web.security import ContentSecurityPolicyMiddleware, add_security_headers
from portolan.web.templating import build_templates

__all__ = ["create_app"]

STATIC_DIR = Path(__file__).parent / "static"

# Statuses rendered as the branded error.html for a browser. 302 stays a
# real redirect (`require_session`); 401 is handled separately in
# `_http_exception_handler` (a /login redirect for a page navigation).
_HTML_ERROR_STATUSES = frozenset({400, 403, 404, 500})


async def _send_mcp_unavailable(send: Any) -> None:
    """The 503 `/mcp` answers with when the MCP transport did not start
    (`_MCPGateway`). This process does not retry the transport; only a
    restart does."""
    body = b'{"error":"unavailable","detail":"the MCP transport is not running"}'
    await send(
        {
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"retry-after", b"30"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class _MCPGateway:
    """The ASGI endpoint registered as `/mcp` at `create_app` time, before
    the MCP transport exists. `lifespan` builds the transport and calls
    `set_app` only after it has started; until then (or if it failed to
    start) every `/mcp` request gets a 503, and the rest of the app still
    boots and serves.
    """

    def __init__(self) -> None:
        self._app: Any | None = None

    def set_app(self, app: Any | None) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        app = self._app
        if app is not None:
            await app(scope, receive, send)
            return
        await _send_mcp_unavailable(send)


def _register_mcp_route(
    app: FastAPI, cfg: Settings, oauth: OAuthResourceConfig | None
) -> _MCPGateway | None:
    """Register `/mcp` when `cfg.mcp_org_key` is set or OAuth is configured,
    else return `None` and register nothing. With OAuth, the public
    protected-resource metadata paths are registered too. Routing only: the
    MCP server itself is built later, in `lifespan`.

    A `Route`, not `app.mount`: Starlette's `Mount` only matches `/mcp/...`,
    so a bare `POST /mcp` would be 307-redirected. A `Route` with an ASGI
    endpoint matches the exact path for every method (`methods` stays
    `None`; see `starlette.routing.Route.__init__`).
    """
    if not cfg.mcp_org_key and oauth is None:
        return None
    gateway = _MCPGateway()
    app.router.routes.append(
        _ASGIRoute("/mcp", endpoint=gateway, include_in_schema=False, name="mcp")
    )
    if oauth is not None:
        metadata_app = ProtectedResourceMetadataApp(oauth)
        for path in oauth.metadata_paths:
            app.router.routes.append(
                _ASGIRoute(path, endpoint=metadata_app, include_in_schema=False)
            )
    return gateway


def _require_auth_settings(cfg: Settings) -> None:
    if cfg.web_auth == "workos":
        missing = [SESSION_SECRET_ENV_VAR] if not cfg.session_secret else []
        missing += signin.validate_config(cfg)
        if missing:
            raise RuntimeError(
                "Portolan web app refuses to start: PORTOLAN_WEB_AUTH=workos but "
                + ", ".join(missing)
                + " not set. Set them in the environment, or set PORTOLAN_WEB_AUTH=none "
                "to use the shared org key."
            )
        return
    checks = (
        (SESSION_SECRET_ENV_VAR, cfg.session_secret),
        (ORG_KEY_ENV_VAR, cfg.org_key),
    )
    missing = [name for name, value in checks if not value]
    if missing:
        raise RuntimeError(
            "Portolan web app refuses to start: "
            + " and ".join(missing)
            + " not set. Without a real session secret, cookies would be forgeable; "
            "without a real org key, there is no key a reader could log in with -- "
            "either way, the whole graph would be readable by anyone who knows this "
            "code exists. Set both in the environment or .env before running the web app."
        )


def _wants_json_error(request: Request) -> bool:
    """True when an error response should stay plain JSON rather than the
    branded `error.html` page:

    - an HTMX request: a failed partial swap should not drop a whole
      document into the fragment target.
    - any path under `/chat/` (the SSE endpoints `/chat/{id}/stream` and
      `/chat/{id}/regenerate`): `fetch()` calls whose JS shows the
      response body inline.
    """
    if request.headers.get("hx-request", "").lower() == "true":
        return True
    return request.url.path.startswith("/chat/")


def _prefers_html_page(request: Request) -> bool:
    """True when the `Accept` header lists `text/html`, as a browser page
    navigation does; API, fetch and httpx callers send `*/*` or nothing.
    Separates a browser that should be redirected to `/login` on a 401
    from a script that should get JSON (the 401 tests in
    `tests/test_web.py` send no `Accept`)."""
    return "text/html" in request.headers.get("accept", "")


def _error_copy(status_code: int, detail: str) -> dict[str, str]:
    """Eyebrow/heading/sentence copy for `error.html`, per status code
    (`04-brand/BRAND.md` voice rule 3: say what actually happened)."""
    if status_code == 404:
        return {
            "eyebrow": "Not found",
            "heading": "That page doesn't exist.",
            "message": "Nothing in the graph matches what was asked for -- it may have "
            "been renamed, removed, or never existed.",
        }
    if status_code == 403:
        return {
            "eyebrow": "Not allowed",
            "heading": "That action isn't allowed.",
            "message": "The request was understood but refused.",
        }
    if status_code == 400:
        return {
            "eyebrow": "Bad request",
            "heading": "That request doesn't work.",
            "message": detail or "The request was malformed or out of range.",
        }
    return {
        "eyebrow": "Something went wrong",
        "heading": "Something went wrong on Portolan's end.",
        "message": "The request failed unexpectedly. Nothing was recorded from it -- try "
        "again, or start from search.",
    }


def create_app(
    settings: Settings | None = None, *, workos_provider: signin.WorkOSProvider | None = None
) -> FastAPI:
    """`workos_provider` replaces the WorkOS SDK client (tests and local proof runs)."""
    cfg = settings or get_settings()
    _require_auth_settings(cfg)

    # Which models are in use, from the process log. "uvicorn.error" because
    # uvicorn's default logging config gives that logger an INFO handler.
    logging.getLogger("uvicorn.error").info(
        "portolan starting: llm_provider=%s extraction_model=%s chat_model=%s",
        cfg.llm_provider,
        cfg.extraction_model,
        chat.resolve_chat_model(cfg),
    )

    # Assigned below by `_register_mcp_route`. `lifespan` closes over this
    # variable and only runs once serving starts, after the assignment.
    gateway: _MCPGateway | None = None
    oauth_config = load_oauth_config(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        startup_log = logging.getLogger("uvicorn.error")

        # Before any migration: the pre-migration backup is written under store_root, so a
        # volume that failed to mount must stop startup here, not after the dump has gone
        # to the container's own disk (tests/test_web_ops.py pins the order).
        assert_store_root_ready(
            cfg.store_root, railway_volume_mount_path=os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
        )

        # Migrations on their own connection, under an advisory lock and
        # after a backup when anything is pending (`db.migrate_with_backup`).
        # A failure raises and the app does not start.
        database_url = cfg.database_url.get_secret_value()
        migration_conn = db.connect(database_url)
        try:
            applied = db.migrate_with_backup(
                migration_conn, database_url=database_url, store_root=cfg.store_root
            )
        finally:
            migration_conn.close()
        if applied:
            startup_log.info(
                "portolan: applied %d migration(s) at startup: %s", len(applied), applied
            )
        else:
            startup_log.info("portolan: no pending migrations at startup")

        # `check` replaces a dead connection before handing it out, so a
        # Postgres restart heals on the next request.
        pool = ConnectionPool(
            database_url,
            min_size=cfg.db_pool_min_size,
            max_size=cfg.db_pool_max_size,
            kwargs={"autocommit": True},
            check=ConnectionPool.check_connection,
            open=False,
        )
        pool.open(wait=True, timeout=30)

        # A fresh deploy gets the company-context note; never overwrites an
        # existing one (`seed_company_context`).
        with pool.connection() as conn:
            seed_company_context(conn)
            # Retention for the pilot activity log (`Settings.activity_retention_days`).
            purged = purge_expired(conn, retention_days=cfg.activity_retention_days)
        startup_log.info("portolan: activity retention purge: %s", purged)

        app.state.settings = cfg
        app.state.pool = pool
        app.state.embedder = get_embedder(cfg)
        app.state.llm = get_llm(cfg)
        app.state.store = LocalStore(cfg.store_root)

        # The MCP server is built fresh on every lifespan cycle because
        # `session_manager.run()` may be entered only once per instance (MCP
        # SDK). A failure here is logged, not raised: `/mcp` answers 503 and
        # the rest of the app, `/healthz` included, still starts.
        # `set_app(None)` on exit stops forwarding to a torn-down transport.
        try:
            async with AsyncExitStack() as mcp_stack:
                if gateway is not None:
                    try:
                        hosted_mcp_server = create_mcp_server(
                            cfg, embedder=app.state.embedder, store=app.state.store, hosted=True
                        )
                        mcp_asgi_app = build_mcp_http_app(
                            hosted_mcp_server,
                            org_key=secret_value(cfg.mcp_org_key),
                            oauth=oauth_config,
                            streamable_http_path="/mcp",
                        )
                        await mcp_stack.enter_async_context(hosted_mcp_server.session_manager.run())
                    except Exception:
                        startup_log.exception(
                            "portolan: MCP transport failed to start; /mcp will return 503 "
                            "for the life of this process rather than stopping the web app "
                            "from booting"
                        )
                    else:
                        gateway.set_app(mcp_asgi_app)

                try:
                    yield
                finally:
                    pool.close()
        finally:
            if gateway is not None:
                gateway.set_app(None)

    # FastAPI's own docs routes carry no auth, so they are off here;
    # `_mount_gated_docs` serves them behind a session when
    # `PORTOLAN_ENABLE_API_DOCS` is set.
    app = FastAPI(
        title="Portolan", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.templates = build_templates()
    app.state.workos_provider = None
    if cfg.web_auth == "workos":
        app.state.workos_provider = workos_provider or signin.build_provider(cfg)

    for module in (
        health,
        login,
        chat,
        notes,
        sources,
        originals,
        graph,
        reports,
        coverage,
        review,
        actions,
        atlas,
        search,
        how_it_works,
        usage,
        people,
    ):
        app.include_router(module.router)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    if cfg.enable_api_docs:
        _mount_gated_docs(app)

    # Routing only; `lifespan` builds the transport behind the gateway.
    gateway = _register_mcp_route(app, cfg, oauth_config)

    _register_error_handlers(app)

    if cfg.web_auth == "workos":
        app.add_middleware(signin.SignInGateMiddleware, settings=cfg)

    # Outermost user middleware, so HTTPException error pages get the CSP
    # too. The unhandled-exception 500 is served from ServerErrorMiddleware,
    # outside every user middleware; its handler adds the headers itself.
    app.add_middleware(ContentSecurityPolicyMiddleware)

    return app


def _render_error_page(request: Request, status_code: int, detail: str) -> Response:
    templates = request.app.state.templates
    copy = _error_copy(status_code, detail)
    context = {
        "detail": detail if status_code in (404, 400) else None,
        **copy,
    }
    return templates.TemplateResponse(request, "error.html", context, status_code=status_code)


def _register_error_handlers(app: FastAPI) -> None:
    """Browser-facing errors render the branded `error.html`: every
    `HTTPException` in `_HTML_ERROR_STATUSES` and any uncaught exception
    (as a 500, never showing exception text). `_wants_json_error` callers
    (HTMX, `/chat/`) get JSON instead.

    A 401 redirects to `/login?next=<path>` when the request is not a
    `_wants_json_error` call and `_prefers_html_page` (a plain form POST,
    such as "+ New chat", can hit an expired session); every other caller
    gets JSON.
    """

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
        is_html_401 = (
            exc.status_code == 401
            and not _wants_json_error(request)
            and _prefers_html_page(request)
        )
        if is_html_401:
            next_path = quote(request.url.path, safe="/")
            return RedirectResponse(url=f"/login?next={next_path}", status_code=303)
        if exc.status_code not in _HTML_ERROR_STATUSES or _wants_json_error(request):
            return JSONResponse(
                {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
            )
        return _render_error_page(request, exc.status_code, str(exc.detail))

    # FastAPI's HTTPException subclasses Starlette's, so this only matters
    # if that ever changes.
    if FastAPIHTTPException is not StarletteHTTPException:  # pragma: no cover - defensive
        app.exception_handler(FastAPIHTTPException)(_http_exception_handler)

    # A malformed path or query parameter (e.g. `GET /c/abc`) gets the
    # branded 400 page in a browser; `_wants_json_error` callers get the
    # structured 422 body.
    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> Response:
        if _wants_json_error(request):
            return JSONResponse({"detail": exc.errors()}, status_code=422)
        return _render_error_page(
            request, 400, "That request doesn't look right -- check the address and try again."
        )

    # Runs inside Starlette's ServerErrorMiddleware, which the CSP middleware
    # never sees, so the security headers are added here. The nonce is the
    # one the middleware already put on `request.state` for the templates.
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> Response:
        del exc  # never rendered -- no traceback or exception text ever reaches a response
        if _wants_json_error(request):
            response: Response = JSONResponse({"detail": "Internal Server Error"}, status_code=500)
        else:
            response = _render_error_page(request, 500, "")
        nonce = getattr(request.state, "csp_nonce", None) or secrets.token_urlsafe(18)
        add_security_headers(response.headers, path=request.url.path, nonce=nonce)
        return response


def _mount_gated_docs(app: FastAPI) -> None:
    """`/docs`, `/redoc`, `/openapi.json`, each behind `require_session`;
    mounted only when `PORTOLAN_ENABLE_API_DOCS` is set."""

    @app.get("/openapi.json", include_in_schema=False)
    def _openapi_json(session: Session = Depends(require_session)) -> JSONResponse:
        del session
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False)
    def _swagger_docs(session: Session = Depends(require_session)) -> object:
        del session
        return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger UI")

    @app.get("/redoc", include_in_schema=False)
    def _redoc_docs(session: Session = Depends(require_session)) -> object:
        del session
        return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")
