"""Signed links for the Monday email: an open pixel and a click redirect.

The email builder calls `email_open_url` for one `<img>` per message and
`email_click_url` for each link it wants counted. The routes that answer
them are in `portolan/web/routes/usage.py` (`/t/o/{token}.gif`,
`/t/c/{token}`); neither needs a login, so the token carries everything and
is signed with `Settings.session_secret`. Rotating that secret stops old
links from counting (a click still lands on `/`).

`recipient` becomes the event's `user_name`: pass the person's login display
name when it is known, so `python -m portolan.activity delete-user` finds
their email rows too. The click target is inside the signature, so the
redirect route can only send a reader to a URL Portolan signed.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from portolan.config import Settings

__all__ = [
    "EMAIL_LINK_MAX_AGE_SECONDS",
    "email_click_url",
    "email_open_url",
    "read_email_token",
]

_SALT = "portolan-email-link-v1"
# Opens and clicks on an email older than this are not counted.
EMAIL_LINK_MAX_AGE_SECONDS = 60 * 60 * 24 * 90
_MAX_LABEL_CHARS = 200


def _serializer(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt=_SALT)


def _secret(settings: Settings) -> str:
    if settings.session_secret is None:
        raise ValueError("PORTOLAN_SESSION_SECRET is not set; email links cannot be signed")
    return settings.session_secret.get_secret_value()


def _base(settings: Settings, base_url: str | None) -> str:
    base = base_url or settings.public_base_url
    if not base:
        raise ValueError("email links need an absolute base: base_url or PORTOLAN_PUBLIC_URL")
    return base.rstrip("/")


def _label(value: str, name: str) -> str:
    value = value.strip()
    if not value or len(value) > _MAX_LABEL_CHARS:
        raise ValueError(f"{name} must be 1 to {_MAX_LABEL_CHARS} characters")
    return value


def email_open_url(
    settings: Settings, *, email_id: str, recipient: str, base_url: str | None = None
) -> str:
    """Absolute URL of a 1x1 GIF that records one `email_open`."""
    token = _serializer(_secret(settings)).dumps(
        {"k": "open", "e": _label(email_id, "email_id"), "r": _label(recipient, "recipient")}
    )
    return f"{_base(settings, base_url)}/t/o/{token}.gif"


def email_click_url(
    settings: Settings,
    *,
    email_id: str,
    recipient: str,
    target: str,
    base_url: str | None = None,
) -> str:
    """Absolute URL that records one `email_click` and redirects to
    `target` (an http(s) URL, or a path on this site such as
    `/reports/<slug>`)."""
    parts = urlsplit(target)
    is_path = target.startswith("/") and not target.startswith("//") and not parts.scheme
    if not is_path and parts.scheme not in ("http", "https"):
        raise ValueError("target must be an http(s) URL or a path starting with /")
    token = _serializer(_secret(settings)).dumps(
        {
            "k": "click",
            "e": _label(email_id, "email_id"),
            "r": _label(recipient, "recipient"),
            "u": target,
        }
    )
    return f"{_base(settings, base_url)}/t/c/{token}"


def read_email_token(token: str, *, secret: str, kind: str) -> dict[str, Any] | None:
    """The payload of a valid, unexpired token of `kind` (`open` or
    `click`), else None."""
    try:
        data = _serializer(secret).loads(token, max_age=EMAIL_LINK_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(data, dict) or data.get("k") != kind:
        return None
    if not isinstance(data.get("e"), str) or not isinstance(data.get("r"), str):
        return None
    if kind == "click" and not isinstance(data.get("u"), str):
        return None
    return data
