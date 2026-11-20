"""Content-Security-Policy and baseline security headers, as a raw ASGI
middleware.

Raw ASGI, not `BaseHTTPMiddleware`: that class relays the response body
through its own task group, which can delay client-disconnect delivery to
the chat SSE stream. This middleware only edits the `http.response.start`
headers and forwards every other message unchanged.

The nonce: one random value per request, put on `scope["state"]` before
the app runs, so `request.state.csp_nonce` is set when a template renders.
Every inline `<script>` carries `nonce="{{ csp_nonce(request) }}"`
(`templating.py`); an injected script lacks it and the browser refuses it.

Not locked down: `style-src` allows `'unsafe-inline'` (templates carry their
own `<style>` blocks, and no path writes untrusted text into one).
`/how-it-works` gets no CSP: it is a standalone static file with its own
un-nonced inline scripts and `data:` audio, which a policy would break.

`frame-ancestors 'self'` lets `original_pdf.html` frame `/originals/{slug}/raw`
while no other origin can frame the app. `Cache-Control`, `Referrer-Policy`
and `X-Content-Type-Options` are set with `setdefault` on every response
this middleware sees, `/how-it-works` included, so a route's own value wins.
Starlette serves the unhandled-exception 500 from `ServerErrorMiddleware`,
outside this middleware, so `app.py`'s 500 handler calls
`add_security_headers` itself (`tests/test_web.py`
`test_500_responses_carry_the_csp_and_baseline_security_headers`).
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import Receive, Scope, Send

__all__ = ["CSP_EXEMPT_PATHS", "ContentSecurityPolicyMiddleware", "add_security_headers"]

# Standalone static page with un-nonced inline scripts (module docstring).
CSP_EXEMPT_PATHS = frozenset({"/how-it-works"})

# CDN hosts the templates load pinned scripts from (`script-src` covers
# `type="module"` scripts too).
_ALLOWED_SCRIPT_ORIGINS = ("https://cdnjs.cloudflare.com", "https://cdn.jsdelivr.net")


def _policy(nonce: str) -> str:
    script_src = " ".join(("'self'", f"'nonce-{nonce}'", *_ALLOWED_SCRIPT_ORIGINS))
    return "; ".join(
        [
            "default-src 'self'",
            f"script-src {script_src}",
            # No nonce for styles (module docstring, "Not locked down").
            "style-src 'self' 'unsafe-inline'",
            # `data:` for graph.html's film-grain overlay, a developer-written
            # inline SVG background image.
            "img-src 'self' data:",
            "font-src 'self'",
            "connect-src 'self'",
            "media-src 'self'",
            "object-src 'none'",
            "base-uri 'self'",
            # 'self', not 'none': original_pdf.html frames the app's own
            # /originals/{slug}/raw.
            "frame-ancestors 'self'",
            "form-action 'self'",
        ]
    )


def add_security_headers(headers: MutableHeaders, *, path: str, nonce: str) -> None:
    """The CSP (except on `CSP_EXEMPT_PATHS`) and the three baseline headers.
    `setdefault` for the baseline ones: a route's own value (e.g.
    /how-it-works' Cache-Control) wins and is never duplicated."""
    if path not in CSP_EXEMPT_PATHS:
        headers.append("Content-Security-Policy", _policy(nonce))
    headers.setdefault("Cache-Control", "private, no-store")
    headers.setdefault("Referrer-Policy", "same-origin")
    headers.setdefault("X-Content-Type-Options", "nosniff")


class ContentSecurityPolicyMiddleware:
    """One nonce per request as `request.state.csp_nonce`; adds the
    `Content-Security-Policy` header (except on `CSP_EXEMPT_PATHS`) and the
    three baseline headers to each response it forwards."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        nonce = secrets.token_urlsafe(18)
        state: MutableMapping[str, Any] = scope.setdefault("state", {})
        state["csp_nonce"] = nonce
        path = scope.get("path", "")

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                add_security_headers(MutableHeaders(scope=message), path=path, nonce=nonce)
            await send(message)

        await self.app(scope, receive, send_wrapper)
