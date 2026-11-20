"""Jinja2 environment shared by every route module: the templates
directory, the custom filters, and globals (brand assets read once at
startup, `app_version`, `get_session`, `csp_nonce`).
"""

from __future__ import annotations

import tomllib
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit

from fastapi.templating import Jinja2Templates

from portolan.util import clock as clock_module
from portolan.web.auth import get_session
from portolan.web.markdown import linkify_wikilinks, render_chat_sentence_text
from portolan.web.review_presenter import highlight_numerals
from portolan.web.snippets import search_snippet

TEMPLATES_DIR = Path(__file__).parent / "templates"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BRAND_DIR = PROJECT_ROOT / "04-brand"
# Mascot SVGs are inlined, not `<img src>`, so their `currentColor` and
# `var(--pl-brass)` pick up the page's light/dark tokens.
STATIC_DIR = Path(__file__).parent / "static"

# Autoescaping stops attribute break-out, not a `javascript:` scheme, so an
# `href` built from vendor or model data (an RSS item's `link`) must pass
# `_safe_href`.
ALLOWED_HREF_SCHEMES = frozenset({"http", "https", "mailto"})


def _read_text(path: Path) -> str:
    """`path`'s contents, or `""` if it can't be read; `build_templates`
    decides whether an empty read is fatal."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _fmt_date(value: object, fmt: str = "%Y-%m-%d") -> str:
    """A `timestamptz` comes back from psycopg labelled in the database
    session's zone (UTC on Railway and in CI), so an aware datetime is
    converted to the deployment's own zone (`portolan.util.clock`) before
    formatting -- otherwise anything written after 17:00 in Phoenix shows
    tomorrow's date. Plain dates and naive datetimes are formatted as-is."""
    if value is None:
        return ""
    if isinstance(value, datetime) and value.tzinfo is not None:
        value = value.astimezone(clock_module.now().tzinfo)
    if isinstance(value, datetime | date):
        return value.strftime(fmt)
    return str(value)


def _relative_time(value: object) -> str:
    """A coarse "how long ago" label ("2h ago") for the chat sidebar:
    minutes, hours, days, weeks, then a plain date past five weeks.
    Measured against `portolan.util.clock.now()`, the deployment's clock.
    """
    if not isinstance(value, datetime):
        return ""
    now = clock_module.now()
    delta = now - (value if value.tzinfo else value.replace(tzinfo=now.tzinfo))
    seconds = delta.total_seconds()
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = int(minutes // 60)
    if hours < 24:
        return f"{hours}h ago"
    days = int(hours // 24)
    if days < 7:
        return f"{days}d ago"
    weeks = int(days // 7)
    if weeks < 5:
        return f"{weeks}w ago"
    return _fmt_date(value)


def _safe_href(value: object) -> str:
    """A URL from vendor or model data, returned only when its scheme is
    `http`, `https`, or `mailto`; anything else (`javascript:`, `data:`,
    scheme-relative `//host`, unparseable) returns `""`. Templates wrap the
    `<a>` in `{% if value|safe_href %}` and render inert text otherwise.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        scheme = urlsplit(text).scheme.lower()
    except ValueError:
        return ""
    if scheme not in ALLOWED_HREF_SCHEMES:
        return ""
    return text


def _app_version() -> str:
    """`[project].version` from `pyproject.toml` for the page footer, or
    "dev" when the file is missing or malformed."""
    try:
        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        return str(data["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "dev"


def build_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["date"] = _fmt_date
    templates.env.filters["relative_time"] = _relative_time
    templates.env.filters["safe_href"] = _safe_href
    templates.env.filters["search_snippet"] = search_snippet
    # A claim's `claim_text` can carry literal `[[slug]]` markers.
    templates.env.filters["linkify_wikilinks"] = linkify_wikilinks
    templates.env.filters["chat_md"] = render_chat_sentence_text
    templates.env.filters["highlight_numerals"] = highlight_numerals
    # Brand and mascot assets are local files under 04-brand/ and static/,
    # never request, vendor or model data, so templates render them `|safe`.
    # `brand_tokens_css` must be `|safe`: escaped quotes break the CSS
    # (tests/test_web.py asserts no `&#34;` in the <style> block).
    templates.env.globals["wordmark_svg"] = _read_text(BRAND_DIR / "wordmark.svg")
    brand_tokens_css = _read_text(BRAND_DIR / "brand-tokens.css")
    # Fail at startup: without the tokens every page renders unstyled with
    # no error (tests/test_web.py::test_build_templates_raises_when_brand_
    # tokens_css_is_missing_or_empty).
    if not brand_tokens_css:
        raise RuntimeError(
            f"Portolan web app refuses to start: {BRAND_DIR / 'brand-tokens.css'} is missing "
            "or empty. Every page's type stacks, dark mode, and accent colors come from this "
            "file -- without it the app would render, but unstyled, with no error to explain "
            "why. Make sure 04-brand/ shipped with this deployment."
        )
    templates.env.globals["brand_tokens_css"] = brand_tokens_css
    # A missing mascot file renders as nothing; decoration is not fatal.
    templates.env.globals["spyglass_svg"] = _read_text(STATIC_DIR / "spyglass.svg")
    templates.env.globals["spyglass_small_svg"] = _read_text(STATIC_DIR / "spyglass-small.svg")
    templates.env.globals["app_version"] = _app_version()
    # base.html reads the session off `request` itself, so routes need not
    # pass one for the page header.
    templates.env.globals["get_session"] = get_session
    # The per-request nonce set by `ContentSecurityPolicyMiddleware`; empty
    # when a render has no middleware in front of it.
    templates.env.globals["csp_nonce"] = lambda request: getattr(request.state, "csp_nonce", "")
    return templates
