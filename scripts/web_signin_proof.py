"""Drive a local Portolan through WorkOS sign-in with a fake AuthKit, end to end.

    uv run python scripts/web_signin_proof.py [--out var/proof/web-signin-2026-09-25]

No WorkOS account and no paid call: a loopback HTTP server plays AuthKit's
authorize and logout pages, and the app gets a provider whose code exchange
checks PKCE against that server's record. The app itself is the real
`create_app` under uvicorn on another loopback port, with web_auth="workos",
the fake embedder and a canned chat answer, against a throwaway local database
(`portolan_test_signin_proof`, dropped afterwards).

The browser is an httpx client with a cookie jar that follows redirects, so every
hop (app -> AuthKit -> app callback -> app page) crosses a real socket. Steps:
signed-out redirect, sign-in, a chat question, the pilot dashboard, a refused
outsider, logout through AuthKit's logout, and the gate afterwards. Writes
`steps.json` (every hop's URL and status, and each check) and the HTML of the
pages it looked at into `--out`. Exit 0 only when every check passed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import secrets
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import psycopg
import uvicorn

from portolan.config import Settings
from portolan.web import answer as answer_module
from portolan.web.app import create_app
from portolan.web.routes import chat as chat_routes
from portolan.web.signin import SignedInPerson, SignInError

DB_NAME = "portolan_test_signin_proof"
ORG = "org_01PROOFPILOT"
ADMIN = "ada@pilot.example"
PEOPLE = {
    ADMIN: SignedInPerson(
        user_id="user_ada",
        email=ADMIN,
        email_verified=True,
        first_name="Ada",
        last_name="Admin",
        organization_id=ORG,
        session_id="session_ada",
    ),
    "eve@elsewhere.example": SignedInPerson(
        user_id="user_eve",
        email="eve@elsewhere.example",
        email_verified=True,
        first_name="Eve",
        last_name="Outsider",
        organization_id="org_SOMEONE_ELSE",
        session_id="session_eve",
    ),
}


@dataclass
class _Pending:
    person: SignedInPerson
    challenge: str


class FakeAuthKit:
    """AuthKit's two browser-facing pages on loopback. `/authorize` signs in
    whoever `next_person` names and redirects back with a code; `/logout` ends
    the session and redirects to `return_to`."""

    def __init__(self) -> None:
        self.next_person = ADMIN
        self.codes: dict[str, _Pending] = {}
        self.ended_sessions: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's naming
                url = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                if url.path == "/authorize":
                    code = "code_" + secrets.token_hex(8)
                    fake.codes[code] = _Pending(PEOPLE[fake.next_person], query["code_challenge"])
                    target = (
                        query["redirect_uri"]
                        + "?"
                        + urlencode({"code": code, "state": query["state"]})
                    )
                elif url.path == "/logout":
                    fake.ended_sessions.append(query["session_id"])
                    target = query["return_to"]
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(302)
                self.send_header("location", target)
                self.send_header("content-length", "0")
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class FakeProvider:
    """`portolan.web.signin.WorkOSProvider` against `FakeAuthKit`."""

    def __init__(self, authkit: FakeAuthKit) -> None:
        self.authkit = authkit

    def authorization_url(self, *, redirect_uri: str, state: str, code_challenge: str) -> str:
        query = {"redirect_uri": redirect_uri, "state": state, "code_challenge": code_challenge}
        return f"{self.authkit.base}/authorize?{urlencode(query)}"

    def authenticate(
        self, *, code: str, code_verifier: str, organization_id: str | None
    ) -> SignedInPerson:
        pending = self.authkit.codes.pop(code, None)
        if pending is None:
            raise SignInError("workos_BadRequestError")
        digest = hashlib.sha256(code_verifier.encode()).digest()
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != pending.challenge:
            raise SignInError("workos_BadRequestError")
        return pending.person

    def logout_url(self, *, session_id: str, return_to: str | None) -> str:
        query = {"session_id": session_id, "return_to": return_to or ""}
        return f"{self.authkit.base}/logout?{urlencode(query)}"


def _canned_answer(**_kwargs: Any) -> answer_module.Answer:
    """Chat's answer step, canned: the fake LLM has no responses, and this proof is
    about who asked, not what was answered."""
    return answer_module.Answer(
        sentences=[answer_module.Sentence(text="A canned answer for the sign-in proof.")],
        retrieved=[],
        control_ok=True,
        usage={},
    )


def _admin_db(sql: str) -> None:
    with psycopg.connect("postgresql://localhost/postgres", autocommit=True) as conn:
        conn.execute(sql)


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Recorder:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.steps: list[dict[str, Any]] = []
        self.failures = 0

    def page(self, name: str, resp: httpx.Response) -> None:
        (self.out / f"{name}.html").write_text(resp.text, encoding="utf-8")

    def step(self, name: str, resp: httpx.Response, checks: dict[str, bool]) -> None:
        hops = [{"status": r.status_code, "url": _scrub(str(r.url))} for r in resp.history]
        hops.append({"status": resp.status_code, "url": _scrub(str(resp.url))})
        passed = all(checks.values())
        self.failures += 0 if passed else 1
        self.steps.append({"step": name, "passed": passed, "hops": hops, "checks": checks})
        print(
            f"{'PASS' if passed else 'FAIL'} {name}: {' -> '.join(str(h['status']) for h in hops)}"
        )


def _scrub(url: str) -> str:
    """Codes, state and PKCE values are one-time and local, but keep them out of the file."""
    return re.sub(r"(code|state|code_challenge)=[^&]+", r"\1=<redacted>", url)


def run(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(out)
    authkit = FakeAuthKit()
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    _admin_db(f"DROP DATABASE IF EXISTS {DB_NAME}")
    _admin_db(f"CREATE DATABASE {DB_NAME}")
    store = Path(tempfile.mkdtemp(prefix="signin-proof-store-"))
    settings = Settings(
        _env_file=None,
        database_url=f"postgresql://localhost/{DB_NAME}",
        tenant="proof",
        store_root=store,
        llm_provider="fake",
        embed_provider="fake",
        session_secret=secrets.token_hex(32),
        cookie_secure=False,
        public_base_url=base,
        web_auth="workos",
        workos_client_id="client_01PROOF",
        workos_api_key=secrets.token_hex(16),
        web_auth_org_id=ORG,
        admin_emails=ADMIN,
    )
    app = create_app(settings, workos_provider=FakeProvider(authkit))
    chat_routes.answer_question = lambda *a, **k: _canned_answer(**k)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(300):
            if server.started:
                break
            time.sleep(0.1)
        html = {"Accept": "text/html"}
        with httpx.Client(base_url=base, follow_redirects=True, timeout=30) as browser:
            home = browser.get("/", headers=html)
            rec.page("01-home-after-sign-in", home)
            rec.step(
                "signed-out visit goes through AuthKit and lands signed in",
                home,
                {
                    "went to /login": any("/login" in str(r.url) for r in home.history),
                    "went to AuthKit authorize": any(
                        str(r.url).startswith(f"{authkit.base}/authorize") for r in home.history
                    ),
                    "came back through /auth/callback": any(
                        "/auth/callback" in str(r.url) for r in home.history
                    ),
                    "ends on / with 200": home.status_code == 200 and home.url.path == "/",
                    "nav shows the verified name": "Ada Admin" in home.text,
                },
            )

            chat = browser.post("/chat/new/stream", data={"question": "What did Ashgrove say?"})
            (out / "02-chat-stream.txt").write_text(chat.text, encoding="utf-8")
            rec.step(
                "a chat question as the signed-in person",
                chat,
                {"200": chat.status_code == 200, "stream finished": "event: done" in chat.text},
            )

            dash = browser.get("/admin/usage", headers=html)
            rec.page("03-admin-usage", dash)
            rec.step(
                "the pilot dashboard names the verified person",
                dash,
                {
                    "200": dash.status_code == 200,
                    "lists Ada Admin <email>": "Ada Admin &lt;ada@pilot.example&gt;" in dash.text,
                    "lists the question": "What did Ashgrove say?" in dash.text,
                },
            )

            people = browser.get("/admin/people", headers=html)
            rec.page("04-admin-people", people)
            rec.step(
                "the people page lists the sign-in as an editor",
                people,
                {
                    "200": people.status_code == 200,
                    "row for ada": ADMIN in people.text,
                    "editor": '<td class="role">editor</td>' in people.text,
                },
            )

            logout = browser.post("/logout")
            rec.page("05-signing-out", logout)
            finish = re.search(r'content="0;url=([^"]+)"', logout.text)
            logout_url = finish.group(1).replace("&amp;", "&") if finish else ""
            rec.step(
                "logout clears the cookie and points at AuthKit's logout",
                logout,
                {
                    "200": logout.status_code == 200,
                    "session cookie gone": "portolan_session" not in browser.cookies,
                    "forwards to AuthKit logout": logout_url.startswith(f"{authkit.base}/logout"),
                },
            )
            signed_out = browser.get(logout_url, headers=html)
            rec.page("06-signed-out", signed_out)
            rec.step(
                "AuthKit's logout ends the WorkOS session and returns to /signed-out",
                signed_out,
                {
                    "ends on /signed-out": signed_out.url.path == "/signed-out",
                    "WorkOS session ended": authkit.ended_sessions == ["session_ada"],
                },
            )

            gated = browser.get("/atlas", headers=html, follow_redirects=False)
            api = browser.get("/graph.json", follow_redirects=False)
            rec.step(
                "after logout every page is gated again",
                gated,
                {
                    "page redirects to login": gated.status_code == 303
                    and gated.headers["location"] == "/login?next=/atlas",
                    "API answers 401": api.status_code == 401,
                },
            )

            authkit.next_person = "eve@elsewhere.example"
            refused = browser.get("/", headers=html)
            rec.page("07-outsider-refused", refused)
            rec.step(
                "someone outside the organization is refused",
                refused,
                {
                    "403": refused.status_code == 403,
                    "no session": "portolan_session" not in browser.cookies,
                    "explains why": "a member of this Portolan" in refused.text,
                },
            )
            health = browser.get("/healthz")
            rec.step("healthz stays public", health, {"200": health.status_code == 200})
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        authkit.close()
        _admin_db(f"DROP DATABASE IF EXISTS {DB_NAME} WITH (FORCE)")

    result = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "app": "create_app(web_auth='workos') under uvicorn on loopback",
        "authkit": "FakeAuthKit on loopback (no WorkOS account, no network)",
        "passed": rec.failures == 0,
        "steps": rec.steps,
    }
    (out / "steps.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    total = len(rec.steps)
    print(f"{'PASS' if rec.failures == 0 else 'FAIL'} {total - rec.failures}/{total} -> {out}")
    return 0 if rec.failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("var/proof/web-signin-2026-09-25"))
    return run(parser.parse_args().out)


if __name__ == "__main__":
    sys.exit(main())
