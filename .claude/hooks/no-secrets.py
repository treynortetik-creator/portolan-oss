#!/usr/bin/env python3
"""no-secrets: a PreToolUse hook that refuses tool calls which would print a secret, commit
one, or (from an agent worktree) touch the live system.

WHY (CLAUDE.md, "Never print deployment variables", "Never commit secrets", "Agents never
touch the live database and never deploy")
  Two incidents put the live database password into an agent transcript: the 2026-09-12
  seeder and `railway variables --kv` on 2026-09-13. A transcript is not a safe place for a
  value: it is stored, summarised and re-read by other agents. This hook runs before every
  Bash, Monitor, Read and Grep call (`.claude/settings.json`) and exits 2 (block; stderr goes
  back to the agent) when the call would put a secret on stdout or stderr.

WHAT IT BLOCKS
  - reading a secrets file: `.env*` (not `.env.example`), the live keys in
    `var/railway/*key*.txt`, and the Railway CLI token (`~/.railway/config.json`). Bash deny verbs: cat, head, tail,
    less, more, bat, nl, sed, awk, grep, cut, strings, xxd, od, sort, cp, `< file`, inline
    python/node code that opens it, ... Allowed on either: `grep -c` / `grep -q` / `grep -l`,
    `test -f`, `ls`, `stat`, `wc`, writing to it (`>> .env`, `cp x .env`). Sourcing is allowed
    for .env only (sourcing a bare key file would print the key in the "command not found"
    error). Running a script that uses a key (`sh var/ops/mcp_probe.sh`) is fine: its contents
    are not visible here. The Read tool on those paths, and the Grep tool with a `path` or
    `glob` naming them, are refused the same way.
  - a recursive grep (or `rg -u`) over a directory that holds a .env, whatever the pattern,
    unless .env is excluded (`--exclude='.env*'`, `-g '!.env*'`) or the files are narrowed
    (`--include='*.py'`, `-t py`).
  - printing the environment: `printenv`, bare `env` (and `env |`), `export`/`export -p`,
    `declare -x` / `declare -p` / bare `declare`, `typeset -m`, bare `set`, `ps eww`,
    `/proc/*/environ`, the python-dotenv CLI (`dotenv list`, `python -m dotenv`), awk's
    ENVIRON, jq's `env`, `docker inspect` without a format, `git credential fill`.
  - `railway variables` unless it is `--kv` piped into `cut -d= -f1` (names only). The command
    after `railway run` or `railway ssh` is checked like a local one.
  - echo / printf of a variable whose name looks secret (KEY, TOKEN, SECRET, PASSWORD, PGPASS,
    DATABASE_*URL, DSN), or of a variable assigned from one (`K=$OPENROUTER_API_KEY; echo $K`).
    `${#NAME}` and `${NAME:+x}` are fine: they print a length or a flag.
  - literals shaped like an AWS, OpenRouter, Anthropic, OpenAI or GitHub key.
  - `set -x` in a command that also sources a .env (xtrace prints every assignment), and
    pytest in a command that sources .env (the suite must never see it; CLAUDE.md "Run it").
  - from an agent worktree (`cwd` under `.claude/worktrees/`): any `railway` command,
    `git push`, `gh pr merge`, running a `var/ops/` script, and the live database host
    (`rlwy.net`). The main checkout keeps all of them.
  - from an agent worktree: any Read, Grep or shell command naming `evals/locked/` (the chat
    exam's test half and its transcripts, `var/evals/locked/` included), except listing it
    (`ls`, `stat`, `wc`, `du`, `test`) and `git add/rm/mv/status/ls-files/commit`. The exam's
    runner reads it in-process and prints only aggregate scores and pass/fail ids.
  - `git push` from the main checkout unless `scripts/ci_local.py` left a passing stamp for
    HEAD's tree on a clean worktree (`push_gate.py`, the same check as the git `pre-push` hook).
  - `git commit` when gitleaks (the repo's `.gitleaks.toml` rules) finds a secret in any change
    the commit could include: staged, unstaged, or untracked and not ignored. The finding is
    reported as rule, file and line only.

HOW
  The command is lexed like a shell would (quotes, escapes, `$(...)`, backticks, heredocs,
  `bash -c`, `eval`, `sudo`/`env`/`uv run`/`railway run` wrappers) and each simple command is
  checked on its own, so a newline, a `;` or a quoted `.e""nv` does not hide the dangerous
  half. Flags are matched as whole arguments, never as substrings of a path.

WHAT IT IS NOT
  A sandbox. It reads command TEXT. A script that reads .env and prints it passes, and so does
  anything built from pieces at run time. It is a tripwire for the accidents that already
  happened. A payload it cannot parse exits 0. An internal error, or analysis that runs past
  its time budget, blocks only when the raw text names a secrets file, a secret-looking
  variable, an environment dump, railway or a push; anything else exits 0, so a bug here never
  bricks the agent. If gitleaks is missing or its scan errors, the commit goes ahead (CI's
  `secrets` job is the backstop).

Runs on the system python3 (3.9 on macOS), so no 3.10+ syntax.
"""

from __future__ import annotations

import fnmatch
import itertools
import json
import os
import re
import signal
import sys

MAX_DEPTH = 8
ANALYSIS_BUDGET_SECONDS = 10

SECRET_NAME = re.compile(
    r"KEY|TOKEN(?!S\b)|SECRET|PASSW(?:OR)?D|PGPASS|DATABASE_\w*URL|DSN", re.I
)

KEY_LITERALS = [
    (re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"), "an AWS access key"),
    (re.compile(r"sk-or-(?:v\d+-)?[A-Za-z0-9]{16,}"), "an OpenRouter key"),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "an Anthropic key"),
    (re.compile(r"sk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}"), "an OpenAI key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}|\bgithub_pat_[A-Za-z0-9_]{22,}"), "a GitHub token"),
]

RULE_PRINT = "Never print deployment variables"
RULE_LIVE = "Agents never touch the live database and never deploy"
RULE_COMMIT = "Never commit secrets"
RULE_LOCKED = "Exam test answers stay out of agents' reach"
RULE_PUSH = "Push only after the local CI gate passes"

# An agent's own checkout: `.claude/worktrees/<name>` (hooks doc: the payload's `cwd` is the
# worktree root once Claude works there, while $CLAUDE_PROJECT_DIR stays at the main checkout).
_WORKTREE = re.compile(r"(?:^|/)\.claude/worktrees/[^/]+")
_VAR_OPS = re.compile(r"(?:^|/)var/ops(?:/|$)")
_LIVE_HOST = re.compile(r"rlwy\.net\b|\.railway\.internal\b")
# The chat exam's locked half (01-spec/chat-exam-2026-09-29.md), in the repo or under var/.
_LOCKED = re.compile(r"(?<![\w.-])evals/locked(?![\w.-])")
_LOCKED_LISTERS = {"ls", "stat", "wc", "du", "test", "["}
_LOCKED_GIT_OK = {"add", "rm", "mv", "status", "ls-files", "commit"}
_LOCKED_GIT_READERS = {"show", "diff", "log", "cat-file", "grep", "blame", "archive"}

ADVICE_DOTENV = (
    "check a key exists with `grep -q '^NAME=' .env && echo present`, count with "
    "`grep -c . .env`, or load it into one process with `set -a; . ./.env; set +a; <command>` "
    "(never printed). For the OpenRouter key: `uv run python scripts/ops/key_status.py`. For "
    "var/railway/*key*.txt: check with `wc -c <file>`, and use it through the script that "
    "reads it (`sh var/ops/mcp_probe.sh`)."
)
ADVICE_ENVDUMP = (
    "test one variable with `[ -n \"${NAME+x}\" ] && echo set`, or list names only with "
    "`python3 -c 'import os; print(sorted(os.environ))'`."
)
ADVICE_RAILWAY = (
    "list names with `railway variables --kv | cut -d= -f1`; use a value without printing it "
    "inside `railway run --service <svc> -- sh -c '...'`."
)
ADVICE_ECHO = (
    "check presence with `[ -n \"${NAME:+x}\" ] && echo set`, or the length with "
    "`echo ${#NAME}`."
)
ADVICE_LITERAL = (
    "keep the value in .env and reference the variable NAME. If a real key was pasted into "
    "this command, rotate it."
)
ADVICE_INLINE = (
    "read the value inside a script that never prints it (scripts/ops/key_status.py is the "
    "pattern), or print only `'NAME' in os.environ`."
)
ADVICE_GREP_TREE = (
    "search code with `rg PATTERN` (skips gitignored .env), name the directories "
    "(`grep -rn PATTERN portolan/ scripts/`), or add `--exclude='.env*'`."
)
ADVICE_LIVE = (
    "agents work in their worktree and never touch the live database or deploy. Hand this "
    "step to the orchestrator (the main session) with the exact command and why."
)
ADVICE_PYTEST = (
    "run the suite with the CLAUDE.md command: `env -u OPENROUTER_API_KEY -u DATABASE_URL "
    "TZ=UTC PGTZ=UTC PORTOLAN_ROLE=test uv run pytest -q <paths>`, never with .env loaded."
)
ADVICE_COMMIT = (
    "take the value out of the file (keep it in .env and reference the variable name), then "
    "commit again. If a real key was written, rotate it. A false positive gets a narrow "
    "allowlist entry in .gitleaks.toml that names where it was found."
)
ADVICE_PUSH = (
    "run `uv run python scripts/ci_local.py` from a clean worktree (commit first), then push "
    "again. GitHub CI is off, so this stamp is the only check before Railway deploys."
)
ADVICE_LOCKED = (
    "the chat exam's test half is out of agents' reach. Run `uv run python -m portolan.bench "
    "chat-exam run --split test ...`: it reads the locked cases in-process and prints aggregate "
    "scores and pass/fail ids only. Train cases and transcripts are readable "
    "(evals/chat-exam/train.json, var/evals/chat-exam/<run>/)."
)
ADVICE_UNCHECKED = (
    "split the command into simpler ones (fewer nested quotes, substitutions or brace groups) "
    "so it can be checked."
)


class Blocked(Exception):
    def __init__(self, reason: str, advice: str, rule: str = RULE_PRINT) -> None:
        super().__init__(reason)
        self.reason = reason
        self.advice = advice
        self.rule = rule


class OverBudget(Exception):
    pass


# --------------------------------------------------------------------------- lexer


class Word:
    """One shell word after quote removal. `refs` are the variables it expands, as
    (name, safe) where safe means the expansion prints a length or a flag, not the value.
    `subs` are the token lists of the command substitutions inside it."""

    def __init__(self) -> None:
        self.parts: list = []
        self.refs: list = []
        self.subs: list = []

    @property
    def text(self) -> str:
        return "".join(self.parts)


class Redir:
    def __init__(self, op: str) -> None:
        self.op = op
        self.target: Word | None = None
        self.heredoc_body: str | None = None
        self.heredoc_quoted = False
        self.dash = False
        self.expanded: Word | None = None  # an unquoted heredoc body, lexed


_REDIR = re.compile(r"(\d*)(<<<|<<-|<<|<>|<&|>>|>&|>\||&>>|&>|<|>)")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ANSI = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "e": "\x1b", "f": "\f",
         "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?"}


def _decode_ansi_c(body: str) -> str:
    out = []
    i = 0
    while i < len(body):
        c = body[i]
        if c != "\\" or i + 1 >= len(body):
            out.append(c)
            i += 1
            continue
        nxt = body[i + 1]
        if nxt in _ANSI:
            out.append(_ANSI[nxt])
            i += 2
        elif nxt == "x":
            m = re.match(r"[0-9A-Fa-f]{1,2}", body[i + 2:])
            if m:
                out.append(chr(int(m.group(0), 16)))
                i += 2 + len(m.group(0))
            else:
                out.append("x")
                i += 2
        elif nxt in "01234567":
            m = re.match(r"[0-7]{1,3}", body[i + 1:])
            out.append(chr(int(m.group(0), 8)))
            i += 1 + len(m.group(0))
        else:
            out.append(nxt)
            i += 2
    return "".join(out)


class Lexer:
    """A shell lexer good enough to find simple commands, their words and redirects.

    Tokens are ("op", str), ("word", Word) or ("redir", Redir)."""

    def __init__(self, s: str, depth: int = 0) -> None:
        self.s = s
        self.i = 0
        self.n = len(s)
        self.depth = depth
        self.pending: list = []

    def lex(self, stop_paren: bool = False) -> list:
        s, toks = self.s, []
        while self.i < self.n:
            c = s[self.i]
            if c in " \t\r":
                self.i += 1
                continue
            if c == "\\" and s.startswith("\n", self.i + 1):
                self.i += 2
                continue
            if c == "\n":
                toks.append(("op", ";"))
                self.i += 1
                self._read_heredocs()
                continue
            if c == "#":
                j = s.find("\n", self.i)
                self.i = self.n if j < 0 else j
                continue
            if c == ")":
                self.i += 1
                if stop_paren:
                    return toks
                toks.append(("op", ")"))
                continue
            if c == "(":
                toks.append(("op", "("))
                self.i += 1
                continue
            if c in "<>" and s.startswith("(", self.i + 1):
                # process substitution <(...) / >(...): a word whose value is a command
                w = Word()
                w.parts.append("/dev/fd/63")
                self.i += 2
                w.subs.append(self._sub_lex())
                toks.append(("word", w))
                continue
            m = _REDIR.match(s, self.i)
            if m:
                self.i = m.end()
                op = m.group(2)
                r = Redir(op)
                self._skip_blanks()
                if self.i < self.n and s[self.i] not in "\n;&|()<>":
                    r.target = self._word()
                if op in ("<<", "<<-") and r.target is not None:
                    raw = s[m.end():self.i]
                    r.heredoc_quoted = any(q in raw for q in "'\"\\")
                    r.dash = op == "<<-"
                    self.pending.append(r)
                toks.append(("redir", r))
                continue
            if c in ";&|":
                two = s[self.i:self.i + 2]
                if two in ("&&", "||", ";;", "|&"):
                    toks.append(("op", two))
                    self.i += 2
                else:
                    toks.append(("op", c))
                    self.i += 1
                continue
            toks.append(("word", self._word()))
        return toks

    def _skip_blanks(self) -> None:
        while self.i < self.n and self.s[self.i] in " \t":
            self.i += 1

    def _read_heredocs(self) -> None:
        s = self.s
        for r in self.pending:
            delim = r.target.text if r.target is not None else ""
            body = []
            while self.i < self.n:
                j = s.find("\n", self.i)
                line = s[self.i:] if j < 0 else s[self.i:j]
                self.i = self.n if j < 0 else j + 1
                if (line.lstrip("\t") if r.dash else line) == delim:
                    break
                body.append(line)
            r.heredoc_body = "\n".join(body)
        self.pending = []

    def _sub_lex(self) -> list:
        """Lex a nested `$(...)` / `<(...)` starting at self.i, up to its closing paren."""
        if self.depth >= MAX_DEPTH:
            # too deep to follow: skip to the next ')' and give up on the inside
            j = self.s.find(")", self.i)
            self.i = self.n if j < 0 else j + 1
            return []
        sub = Lexer(self.s, self.depth + 1)
        sub.i = self.i
        toks = sub.lex(stop_paren=True)
        if sub.pending:
            sub._read_heredocs()
        self.i = sub.i
        return toks

    def _word(self) -> Word:
        s, w = self.s, Word()
        while self.i < self.n:
            c = s[self.i]
            if c in " \t\r\n;&|()<>":
                break
            if c == "\\":
                if self.i + 1 < self.n:
                    if s[self.i + 1] != "\n":
                        w.parts.append(s[self.i + 1])
                    self.i += 2
                else:
                    self.i += 1
                continue
            if c == "'":
                j = s.find("'", self.i + 1)
                j = self.n if j < 0 else j
                w.parts.append(s[self.i + 1:j])
                self.i = j + 1
                continue
            if c == '"':
                self.i += 1
                self._dquote(w, terminator='"')
                continue
            if c == "$":
                self._dollar(w)
                continue
            if c == "`":
                self._backtick(w)
                continue
            w.parts.append(c)
            self.i += 1
        return w

    def _dquote(self, w: Word, terminator: str | None) -> None:
        """Consume a double-quoted body (or, with terminator None, a heredoc body)."""
        s = self.s
        while self.i < self.n:
            c = s[self.i]
            if terminator is not None and c == terminator:
                self.i += 1
                return
            if c == "\\" and self.i + 1 < self.n and s[self.i + 1] in '$`"\\\n':
                if s[self.i + 1] != "\n":
                    w.parts.append(s[self.i + 1])
                self.i += 2
                continue
            if c == "$":
                self._dollar(w)
                continue
            if c == "`":
                self._backtick(w)
                continue
            w.parts.append(c)
            self.i += 1

    def _dollar(self, w: Word) -> None:
        s, i = self.s, self.i
        nxt = s[i + 1] if i + 1 < self.n else ""
        if s.startswith("$((", i):
            depth, j = 0, i + 1
            while j < self.n:
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner = s[i + 3:j]
            for m in re.finditer(r"\$?\{?([A-Za-z_][A-Za-z0-9_]*)", inner):
                w.refs.append((m.group(1), True))
            w.parts.append("0")
            self.i = j + 1
            return
        if nxt == "(":
            self.i = i + 2
            w.subs.append(self._sub_lex())
            w.parts.append("$(...)")
            return
        if nxt == "{":
            depth, j = 0, i + 1
            while j < self.n:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner = s[i + 2:j]
            self._brace_ref(w, inner)
            w.parts.append("${" + inner + "}")
            self.i = j + 1
            return
        if nxt == "'":
            j = i + 2
            while j < self.n and s[j] != "'":
                j += 2 if s[j] == "\\" else 1
            w.parts.append(_decode_ansi_c(s[i + 2:j]))
            self.i = j + 1
            return
        if nxt == '"':
            self.i = i + 2
            self._dquote(w, terminator='"')
            return
        m = _NAME.match(s, i + 1)
        if m:
            w.refs.append((m.group(0), False))
            w.parts.append("$" + m.group(0))
            self.i = m.end()
            return
        w.parts.append("$" + nxt if nxt and nxt in "@*#?$!-0123456789" else "$")
        self.i = i + (2 if nxt and nxt in "@*#?$!-0123456789" else 1)

    def _brace_ref(self, w: Word, inner: str) -> None:
        if inner.startswith("("):
            # zsh parameter flags: ${(P)name} is indirect, like bash's ${!name}
            j = inner.find(")")
            if j > 0:
                if "P" in inner[1:j]:
                    w.refs.append(("INDIRECT_SECRET", False))
                    return
                inner = inner[j + 1:]
        if inner.startswith("#"):
            m = _NAME.match(inner, 1)
            if m and m.end() == len(inner):
                w.refs.append((m.group(0), True))  # ${#NAME}: a length
                return
        if inner.startswith("!"):
            m = _NAME.match(inner, 1)
            if m and inner[m.end():] in ("*", "@"):
                return  # ${!PREFIX*}: variable NAMES
            # ${!ref}: indirect -- the value of whatever ref names; assume the worst
            w.refs.append(("INDIRECT_SECRET", False))
            return
        m = _NAME.match(inner)
        if not m:
            return
        rest = inner[m.end():]
        safe = rest.startswith(":+") or rest.startswith("+")
        w.refs.append((m.group(0), safe))
        for m2 in re.finditer(r"\$\{?#?([A-Za-z_][A-Za-z0-9_]*)", rest):
            w.refs.append((m2.group(1), False))

    def _backtick(self, w: Word) -> None:
        s, j = self.s, self.i + 1
        buf = []
        while j < self.n and s[j] != "`":
            if s[j] == "\\" and j + 1 < self.n and s[j + 1] in "`$\\":
                buf.append(s[j + 1])
                j += 2
                continue
            buf.append(s[j])
            j += 1
        self.i = j + 1
        if self.depth < MAX_DEPTH:
            w.subs.append(Lexer("".join(buf), self.depth + 1).lex_all())
        w.parts.append("`...`")

    def lex_all(self) -> list:
        toks = self.lex()
        if self.pending:
            self._read_heredocs()
        return toks


def lex_heredoc_body(body: str, depth: int) -> Word:
    """An unquoted heredoc body expands like a double-quoted string."""
    lx = Lexer(body, depth)
    w = Word()
    lx._dquote(w, terminator=None)
    return w


# --------------------------------------------------------------------------- parser


class Cmd:
    def __init__(self) -> None:
        self.assigns: list = []
        self.words: list = []
        self.redirs: list = []


_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=")
_PREFIX_KEYWORDS = {"if", "then", "else", "elif", "do", "while", "until", "!", "{", "}",
                    "fi", "done", "esac", "time", "coproc"}


def parse(tokens: list) -> list:
    """Tokens -> pipelines (lists of Cmd)."""
    pipelines, pipe, cmd = [], [], Cmd()

    def end_cmd() -> None:
        nonlocal cmd
        if cmd.words or cmd.redirs or cmd.assigns:
            pipe.append(cmd)
        cmd = Cmd()

    def end_pipe() -> None:
        nonlocal pipe
        end_cmd()
        if pipe:
            pipelines.append(pipe)
        pipe = []

    for kind, val in tokens:
        if kind == "op":
            if val in ("|", "|&"):
                end_cmd()
            else:
                end_pipe()
        elif kind == "redir":
            cmd.redirs.append(val)
        else:
            text = val.text
            if not cmd.words and text in _PREFIX_KEYWORDS and not val.subs:
                continue
            if not cmd.words and _ASSIGN.match(text):
                cmd.assigns.append(val)
                continue
            cmd.words.append(val)
    end_pipe()
    return pipelines


# --------------------------------------------------------------------------- analysis

_READERS_OK_ON_DOTENV = {
    "test", "[", "[[", "ls", "stat", "file", "touch", "chmod", "chown", "wc", "du",
    "realpath", "readlink", "basename", "dirname", "echo", "printf", "rm", "tee", "true",
    "false", ":", "mkdir", "cd", "pushd",
}
_GREPS = {"grep", "egrep", "fgrep", "rg", "ggrep"}
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "fish"}
_PYTHONS = re.compile(r"^(python|python3(\.\d+)?|pypy3?)$")
_OTHER_INTERPRETERS = {"node", "deno", "bun", "ruby", "perl", "php"}
_PRINTERS = {"cat", "tee", "head", "tail", "less", "more", "bat"}
_HTTP = {"curl", "wget", "http", "https", "xh"}
_AWKS = {"awk", "gawk", "mawk", "nawk"}
_JQS = {"jq", "gojq", "jaq"}
_PKG_RUNNERS = {"npx", "bunx", "pnpx"}
# Commands that only look at a file: naming a var/ops script to them does not run it.
_VIEWERS = {
    "cat", "head", "tail", "less", "more", "bat", "ls", "wc", "grep", "egrep", "fgrep", "rg",
    "diff", "stat", "file", "nl", "git", "test", "[", "[[", "echo", "printf",
}
_DOTENV_BASENAME = re.compile(r"^(\.env[^\s/]*|[^\s/]+\.env)$")
_DOTENV_SAMPLES = (".env", ".env.local", ".envrc", "prod.env")
# Only the Railway CLI token: it deploys. Other home credential files (~/.docker/config.json,
# ~/.aws/credentials, ~/.netrc, ...) are readable by choice (the maintainer, 2026-09-29).
_HOME_SECRETS = ((".railway", "config.json"),)
_BRACE = re.compile(r"\{([^{}]*,[^{}]*)\}")
_WALK_SKIP = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache",
              ".pytest_cache"}


def cmdname(text: str) -> str:
    return text.rsplit("/", 1)[-1]


def _brace_once(text: str, limit: int) -> list:
    groups = list(_BRACE.finditer(text))
    if not groups:
        return [text]
    pieces, alts, last = [], [], 0
    for m in groups:
        pieces.append(text[last:m.start()])
        alts.append(m.group(1).split(","))
        last = m.end()
    tail = text[last:]
    if len(groups) <= 8:
        combos = list(itertools.islice(itertools.product(*alts), limit))
    else:
        # Too many groups to multiply out: vary one group at a time (linear, never exponential).
        first = [a[0] for a in alts]
        combos = [tuple(first)]
        for gi, a in enumerate(alts):
            for alt in a[1:]:
                if len(combos) >= limit:
                    break
                c = list(first)
                c[gi] = alt
                combos.append(tuple(c))
    return ["".join(p + c for p, c in zip(pieces, combo)) + tail for combo in combos]


def _brace_expand(text: str, limit: int = 64) -> list:
    cands = [text]
    for _ in range(4):  # nested groups expand from the inside out
        if not any(_BRACE.search(c) for c in cands):
            break
        nxt = []
        for c in cands:
            nxt.extend(_brace_once(c, limit))
        cands = nxt[:limit]
    return cands


def _glob_hits(pattern: str, samples) -> bool:
    return any(fnmatch.fnmatchcase(s, pattern) for s in samples)


def is_dotenv_path(text: str) -> bool:
    """A `.env*` file (not `.env.example`), a `name.env`, or a glob that reaches one."""
    for cand in _brace_expand(text):
        base = cand.rstrip("/").rsplit("/", 1)[-1]
        if base.endswith(".env.example"):
            continue
        if any(ch in base for ch in "*?["):
            if (base.startswith(".") or "env" in base) and _glob_hits(base, _DOTENV_SAMPLES):
                return True
            continue
        if _DOTENV_BASENAME.match(base):
            return True
    return False


_KEYFILE_SAMPLES = ("org-key.txt", "mcp-key.txt", "api-key.txt")


def _component_hits(component: str, names: tuple) -> bool:
    if any(ch in component for ch in "*?["):
        return any(fnmatch.fnmatchcase(n, component) for n in names)
    return component in names


def is_keyfile_path(text: str, in_railway_dir: bool = False) -> bool:
    """`var/railway/*key*.txt` (the live org and MCP keys), a glob that reaches one, or the
    `var/railway` directory itself (`grep -r . var/railway`, `cp -r var/railway /tmp`)."""
    for cand in _brace_expand(text):
        parts = [p for p in cand.split("/") if p not in ("", ".")]
        if not parts:
            continue
        base = parts[-1]
        parent = parts[-2] if len(parts) >= 2 else None
        if any(ch in base for ch in "*?["):
            base_is_key = _component_hits(base, _KEYFILE_SAMPLES)
        else:
            base_is_key = fnmatch.fnmatchcase(base, "*key*.txt")
        if base_is_key and parent is not None and _component_hits(parent, ("railway",)):
            return True
        if base_is_key and parent is None and in_railway_dir:
            return True
        if parent is not None and _component_hits(base, ("railway",)) and _component_hits(parent, ("var",)):
            return True
    return False


def is_home_secret_path(text: str) -> bool:
    """~/.railway/config.json, the Railway CLI token (the only credential outside the repo)."""
    for cand in _brace_expand(text):
        parts = [p for p in cand.split("/") if p not in ("", ".")]
        if not parts:
            continue
        base = parts[-1]
        parent = parts[-2] if len(parts) >= 2 else ""
        for want_parent, name in _HOME_SECRETS:
            if _component_hits(base, (name,)) and (not want_parent or _component_hits(parent, (want_parent,))):
                return True
    return False


def is_secret_path(text: str) -> bool:
    return is_dotenv_path(text) or is_keyfile_path(text) or is_home_secret_path(text)


def tokens_mention_dotenv(tokens: list, depth: int = 0) -> bool:
    """True when a substitution's own words name a .env file (`$(echo .env)`), so the word it
    produces is treated as that path."""
    if depth > MAX_DEPTH:
        return False
    for kind, val in tokens:
        if kind == "word":
            if is_secret_path(val.text) or any(tokens_mention_dotenv(t, depth + 1) for t in val.subs):
                return True
        elif kind == "redir" and val.target is not None and is_secret_path(val.target.text):
            return True
    return False


def unsafe_secret_refs(w: Word, tainted=()) -> list:
    return [
        n for n, safe in w.refs
        if not safe and (n == "INDIRECT_SECRET" or SECRET_NAME.search(n) or n in tainted)
    ]


def short_flags(args: list) -> set:
    """Letters of short-option clusters (`-qE` -> {q, E}); stops at `--`."""
    out = set()
    for a in args:
        if a == "--":
            break
        if a.startswith("-") and not a.startswith("--") and len(a) > 1:
            out.update(a[1:])
    return out


def grep_prints_no_lines(args: list) -> bool:
    longs = {a.split("=", 1)[0] for a in args if a.startswith("--")}
    if longs & {"--count", "--quiet", "--silent", "--files-with-matches", "--files-without-match"}:
        return True
    return bool(short_flags(args) & set("cqlL"))


def grep_pattern_index(args: list):
    """Index in `args` of grep's positional PATTERN (None when -e/-f supply it). A pattern is
    text to match, not a file, so `grep -n '\\.env' .gitignore` reads no secrets file."""
    for a in args:
        if a in ("-e", "--regexp", "-f", "--file") or a.startswith(("--regexp=", "--file=")):
            return None
        if a.startswith("-") and not a.startswith("--") and len(a) > 1 and a[-1] in "ef":
            return None
    for i, a in enumerate(args):
        if a == "--":
            return i + 1 if i + 1 < len(args) else None
        if a.startswith("-") and a != "-":
            continue
        return i
    return None


def _dotenv_excluded(args: list, a0: str) -> bool:
    """True when the grep/rg options keep every .env file out of the search."""
    i = 0
    while i < len(args):
        a = args[i]
        val = None
        if a.startswith(("--exclude=", "--include=", "--glob=", "--iglob=", "--type=")):
            opt, val = a.split("=", 1)
        elif a in ("--exclude", "--include", "--glob", "--iglob", "-g", "--type", "-t") and i + 1 < len(args):
            opt, val = a, args[i + 1]
            i += 1
        elif a.startswith(("-g", "-t")) and len(a) > 2 and a0 == "rg":
            opt, val = a[:2], a[2:]
        else:
            i += 1
            continue
        i += 1
        if opt == "--exclude" and _glob_hits(val, (".env",)):
            return True
        if opt in ("-g", "--glob", "--iglob"):
            if val.startswith("!") and _glob_hits(val[1:].rsplit("/", 1)[-1], (".env",)):
                return True
            if not val.startswith("!") and not _glob_hits(val.rsplit("/", 1)[-1], _DOTENV_SAMPLES):
                return True  # only files matching a positive glob are searched
        if opt == "--include" and not _glob_hits(val, _DOTENV_SAMPLES):
            return True
        if opt in ("-t", "--type") and val not in ("all", "env"):
            return True
    return False


def _tree_holds_dotenv(root: str, max_depth: int = 4, budget: int = 20000) -> bool:
    """A .env* file (not .env.example) at most `max_depth` levels down. Over budget counts as
    yes: a tree that big is not one to grep blind."""
    if not os.path.isdir(root):
        return False
    seen = 0
    base = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        seen += len(dirnames) + len(filenames)
        if seen > budget:
            return True
        for name in filenames:
            if is_dotenv_path(name):
                return True
        depth = dirpath.rstrip("/").count("/") - base
        dirnames[:] = [d for d in dirnames if d not in _WALK_SKIP] if depth < max_depth - 1 else []
    return False


# inline interpreter code (python -c, node -e, ...)
_CODE_DOTENV = re.compile(
    r"(?<![\w.\\])\.env(?:rc|\.(?!example\b)[\w-]+)?(?![\w.-])|\bdotenv_values\s*\(|"
    r"railway/[^'\"\s]*key[^'\"\s]*\.txt|\bvar/railway\b"
)
# Terminal output only: writing a file (`open(p, "w").write(...)`) is not printing.
_CODE_OUTPUT = re.compile(
    r"\b(?:print|pprint|puts|say|printf|echo)\b|sys\.std(?:out|err)|/dev/(?:stdout|stderr|tty)|"
    r"console\.\w+|\blog(?:ging|ger)?\.\w+\s*\(|process\.std(?:out|err)"
)
_CODE_SAFE_FORMS = [
    re.compile(r"\bredact_dsn\s*\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)"),
    re.compile(r"""(['"])\w+\1\s+(?:not\s+)?in\s+(?:os\.)?environ\b"""),
    re.compile(r"\b(?:bool|len)\s*\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)"),
    re.compile(r"[\w.\[\]'\"()]+\s+is\s+(?:not\s+)?None"),
    re.compile(r"\b(?:sorted|list|set|tuple)\s*\(\s*(?:os\.)?environ(?:\.keys\(\))?\s*\)"),
    re.compile(r"\benviron\.keys\s*\(\s*\)|\bObject\.keys\s*\(\s*process\.env\s*\)"),
]
# A secret-looking name counts only in code that also reads the environment or settings, so
# `print(d.keys())` or `print(tokens)` in a json one-liner is not a finding.
_CODE_ENV_ACCESS = re.compile(
    r"\benviron\b|\bgetenv\b|\bprocess\.env\b|\bENV\s*[\[.{]|%ENV\b|\$ENV\{|"
    r"\bsettings\b(?!\.json)|\bget_settings\b|\bSettings\s*\(|\bdotenv\b|"
    r"\bconfig\b(?!\.(?:json|ya?ml|toml))"
)
_CODE_SECRET = re.compile(
    r"\w*(?:api_?key|_key\b|\bkey_|secret|token(?!s\b)|passw(?:or)?d|database_\w*url|dsn)\w*", re.I
)
_CODE_WHOLE_ENV = re.compile(
    r"\benviron\b(?!\s*(?:\[|\.get\b|\.setdefault\b|\.pop\b|\.keys\b|\.__contains__\b))|"
    r"\bprocess\.env\b(?!\s*[.\[])|%ENV\b|\bENV\.(?:to_h|inspect|to_a|each)\b|"
    r"(?<![\w$%.\"'])ENV\b(?!\s*[\[.{])|"
    r"\bget_settings\s*\(\s*\)\s*(?:\)|\.\s*model_dump|\.\s*__dict__)|"
    r"\bvars\s*\(\s*get_settings\s*\(\s*\)\s*\)|"
    r"\bSettings\s*\([^)]*\)\s*(?:\)|\.model_dump)"
)
_CODE_SETTINGS_ALIAS = re.compile(r"\b(\w+)\s*=\s*(?:[\w.]+\.)?(?:get_settings|Settings)\s*\(\s*\)")
_CODE_SHELLOUT = re.compile(
    r"\bos\.(?:system|popen|exec\w*|spawn\w*)\b|\bsubprocess\b|\bPopen\b|\bchild_process\b|"
    r"\bexecSync\b|\bspawnSync\b|\bsystem\s*\(|`"
)
_CODE_STRING = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")
_CODE_LIST = re.compile(r"\[([^\[\]]*)\]")


class Analyzer:
    def __init__(self, cwd: str | None = None) -> None:
        self.vars: dict = {}
        self.tainted: set = set()  # variables assigned from a secret (`K=$OPENROUTER_API_KEY`)
        self.sources_dotenv = False
        self.xtrace = False
        self.in_railway_dir = False  # after `cd var/railway`, a bare `org-key.txt` counts
        self.cwd = cwd
        self.curdir = cwd
        self.worktree = bool(cwd) and bool(_WORKTREE.search(cwd.replace("\\", "/")))
        self.commits: list = []  # directories a `git commit` runs in
        self.pushes: list = []  # directories a `git push` runs in

    # --- entry points
    def analyze_text(self, text: str, depth: int) -> None:
        if depth > MAX_DEPTH:
            return
        self.analyze_tokens(Lexer(text, depth).lex_all(), depth)

    def analyze_tokens(self, tokens: list, depth: int) -> None:
        for pipe in parse(tokens):
            self.analyze_pipeline(pipe, depth)

    # --- helpers
    def expand(self, text: str) -> str:
        def sub(m: re.Match) -> str:
            name = m.group(1) or m.group(2)
            return self.vars.get(name, m.group(0))

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", sub, text)

    def bad_refs(self, w: Word) -> list:
        return unsafe_secret_refs(w, self.tainted)

    def dotenv(self, w: Word | None) -> bool:
        """True when the word names a secrets file: a .env, a var/railway key file, or a home
        credential file."""
        return self.secret_kind(w) is not None

    def secret_kind(self, w: Word | None) -> str | None:
        if w is None:
            return None
        text = self.expand(w.text)
        if is_dotenv_path(text):
            return "env"
        if is_keyfile_path(text, self.in_railway_dir) or is_home_secret_path(text):
            return "key"
        if any(tokens_mention_dotenv(sub) for sub in w.subs):
            return "key"  # produced by a substitution: unknown kind, so no sourcing exemption
        return None

    def live(self, what: str) -> Blocked:
        return Blocked(f"{what} from an agent worktree", ADVICE_LIVE, RULE_LIVE)

    # --- pipeline / command
    def analyze_pipeline(self, pipe: list, depth: int) -> None:
        mentioned_dotenv = False
        for idx, cmd in enumerate(pipe):
            info = self.analyze_cmd(cmd, pipe, idx, depth)
            if info.get("xargs") and (mentioned_dotenv or info.get("dotenv")):
                raise Blocked("feeds a secrets file through xargs into another command", ADVICE_DOTENV)
            mentioned_dotenv = mentioned_dotenv or bool(info.get("dotenv"))

    def analyze_cmd(self, cmd: Cmd, pipe: list, idx: int, depth: int) -> dict:
        # substitutions and heredocs run first, wherever they sit
        for w in cmd.assigns + cmd.words:
            for sub in w.subs:
                self.analyze_tokens(sub, depth + 1)
        for r in cmd.redirs:
            if r.target is not None:
                for sub in r.target.subs:
                    self.analyze_tokens(sub, depth + 1)
            if r.op in ("<<", "<<-") and r.heredoc_body and not r.heredoc_quoted:
                body = lex_heredoc_body(r.heredoc_body, depth + 1)
                for sub in body.subs:
                    self.analyze_tokens(sub, depth + 1)
                r.expanded = body

        for w in cmd.assigns:
            self.assign(w)

        for w in cmd.words + cmd.assigns + [r.target for r in cmd.redirs if r.target]:
            if re.search(r"/proc/[^/\s]+/environ\b", w.text):
                raise Blocked("reads a process environment from /proc", ADVICE_ENVDUMP)

        info: dict = {}
        eff, extra, flags = self.unwrap(cmd, depth)
        info["xargs"] = flags.get("xargs", False)
        a0 = cmdname(eff[0].text) if eff else ""
        args = [w.text for w in eff[1:]]

        # which secrets files does this command touch, and how
        file_words = list(eff[1:])
        if a0 in _GREPS:
            pi = grep_pattern_index(args)
            if pi is not None:
                file_words = file_words[:pi] + file_words[pi + 1:]
        arg_refs = [w for w in file_words + extra if self.dotenv(w)]
        # output redirects into .env (`>> .env`) are writes and print nothing
        in_redirs = [r for r in cmd.redirs if r.op in ("<", "<>") and self.dotenv(r.target)]
        info["dotenv"] = bool(arg_refs or in_redirs)

        if arg_refs or in_redirs:
            self.check_dotenv(a0, eff, args, arg_refs, in_redirs)
        if a0 in ("cd", "pushd") and args:
            target = self.expand(args[-1])
            self.in_railway_dir = target.rstrip("/").endswith("railway")
            if self.curdir:
                self.curdir = os.path.normpath(os.path.join(self.curdir, os.path.expanduser(target)))
        if a0 in ("export", "declare", "typeset", "local", "readonly"):
            for w in eff[1:]:
                if _ASSIGN.match(w.text):
                    self.assign(w)

        if self.worktree:
            self.check_worktree(a0, eff)
            self.check_locked(a0, eff, cmd)
        if a0 == "dotenv":
            raise Blocked("runs the python-dotenv CLI, whose `list` and `get` print .env values",
                          ADVICE_DOTENV)
        if a0 in ("pytest", "py.test") and self.sources_dotenv:
            raise Blocked("runs pytest in a command that sourced .env", ADVICE_PYTEST)
        self.check_recursive_grep(a0, args)
        self.check_env_dumps(a0, args, eff)
        self.check_git(a0, args)
        self.check_railway(a0, args, cmd, pipe, idx)
        self.check_echo(a0, eff, args, cmd)
        return info

    def assign(self, w: Word) -> None:
        name = w.text.partition("=")[0].split("[", 1)[0].rstrip("+")
        if self.bad_refs(w):
            self.tainted.add(name)
        elif not w.subs:
            self.vars[name] = self.expand(w.text.split("=", 1)[1])

    def unwrap(self, cmd: Cmd, depth: int):
        """Strip wrappers (sudo, env, uv run, railway run, ...) down to the command that runs.
        Returns (effective words, wrapper option words that still count as file args, flags)."""
        words = list(cmd.words)
        extra: list = []
        flags: dict = {}
        while words:
            a0 = cmdname(words[0].text)
            rest = words[1:]
            if self.worktree and (a0 == "railway" or words[0].text.startswith("@railway/")):
                raise self.live("runs `railway`")
            if a0 in ("sudo", "doas"):
                words = rest
                while words and words[0].text.startswith("-"):
                    opt = words[0].text
                    words = words[2:] if opt in ("-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U") else words[1:]
                continue
            if a0 in ("nohup", "exec", "command", "builtin", "time", "stdbuf", "caffeinate", "unbuffer", "noglob"):
                words = rest
                while words and words[0].text.startswith("-"):
                    words = words[2:] if words[0].text in ("-a",) else words[1:]
                continue
            if a0 in _PKG_RUNNERS:
                words = rest
                while words and words[0].text.startswith("-"):
                    words = words[2:] if words[0].text in ("-p", "--package") else words[1:]
                continue
            if a0 == "nice":
                words = rest
                while words and words[0].text.startswith("-"):
                    words = words[2:] if words[0].text == "-n" else words[1:]
                continue
            if a0 == "timeout":
                words = rest
                while words and words[0].text.startswith("-"):
                    words = words[2:] if words[0].text in ("-s", "-k", "--signal", "--kill-after") else words[1:]
                words = words[1:]  # the duration
                continue
            if a0 == "env":
                words = rest
                split_string = None
                while words:
                    t = words[0].text
                    if t in ("-u", "--unset", "-C", "--chdir", "-P"):
                        words = words[2:]
                    elif t in ("-S", "--split-string"):
                        split_string = words[1].text if len(words) > 1 else ""
                        words = words[2:]
                    elif t.startswith("--split-string="):
                        split_string = t.split("=", 1)[1]
                        words = words[1:]
                    elif t == "--":
                        words = words[1:]
                        break
                    elif t.startswith("-") and len(t) > 1:
                        words = words[1:]
                    elif _ASSIGN.match(t):
                        self.assign(words[0])
                        words = words[1:]
                    else:
                        break
                if split_string is not None:
                    self.analyze_text(split_string, depth + 1)
                    return [], extra, flags
                if not words:
                    raise Blocked("runs `env` with no command, which prints every variable", ADVICE_ENVDUMP)
                continue
            if a0 == "xargs":
                flags["xargs"] = True
                words = rest
                while words and words[0].text.startswith("-"):
                    t = words[0].text
                    if t in ("-I", "-n", "-P", "-L", "-d", "-E", "-s", "-a", "--arg-file", "-J", "-R", "-S"):
                        if len(words) > 1:
                            extra.append(words[1])
                        words = words[2:]
                    else:
                        words = words[1:]
                continue
            if a0 == "uv" and rest and rest[0].text == "run":
                words = rest[1:]
                while words and words[0].text.startswith("-"):
                    t = words[0].text
                    if t == "--":
                        words = words[1:]
                        break
                    if t == "--env-file" and len(words) > 1:
                        self.sources_dotenv = self.sources_dotenv or self.dotenv(words[1])
                        words = words[2:]
                    elif t.startswith("--env-file="):
                        self.sources_dotenv = self.sources_dotenv or is_dotenv_path(t.split("=", 1)[1])
                        words = words[1:]
                    elif t in ("--directory", "--project", "--with", "--python", "-p", "--package",
                               "--extra", "--group", "--index", "--script", "--with-requirements",
                               "--env-file", "--no-env-file", "-w"):
                        words = words[2:] if t not in ("--no-env-file",) else words[1:]
                    else:
                        words = words[1:]
                continue
            if a0 == "railway" and rest and rest[0].text in ("run", "ssh"):
                sub = rest[0].text
                words = rest[1:]
                while words and words[0].text.startswith("-"):
                    t = words[0].text
                    if t == "--":
                        words = words[1:]
                        break
                    takes_value = t in ("-s", "--service", "-e", "--environment", "-p", "--project",
                                        "-d", "--deployment-instance", "-i", "--identity-file")
                    words = words[2:] if takes_value else words[1:]
                if sub == "ssh":
                    # the remote command runs where every deployment variable is set
                    if len(words) == 1 and " " in words[0].text.strip():
                        self.analyze_text(words[0].text, depth + 1)
                        return [], extra, flags
                    if not words:
                        return [], extra, flags  # an interactive shell: nothing printed here
                continue
            if a0 in _SHELLS:
                self.shell(words, cmd, depth)
                return words, extra, flags
            if a0 == "eval":
                self.analyze_text(" ".join(w.text for w in rest), depth + 1)
                return words, extra, flags
            if _PYTHONS.match(a0) or a0 in _OTHER_INTERPRETERS:
                self.interpreter(a0, words, cmd, depth)
                return words, extra, flags
            break
        return words, extra, flags

    def shell(self, words: list, cmd: Cmd, depth: int) -> None:
        opts = []
        i = 1
        code = None
        while i < len(words):
            t = words[i].text
            if t == "-o" and i + 1 < len(words):
                if words[i + 1].text in ("xtrace", "verbose"):
                    self.xtrace = True
                i += 2
                continue
            if t.startswith("-") and len(t) > 1 and not t.startswith("--"):
                opts.append(t)
                if "c" in t[1:]:
                    code = words[i + 1].text if i + 1 < len(words) else ""
                    break
                i += 1
                continue
            break
        if any(set(o[1:]) & set("xv") for o in opts):
            self.xtrace = True
        if code is not None:
            self.analyze_text(code, depth + 1)
        for r in cmd.redirs:
            if r.op in ("<<", "<<-") and r.heredoc_body is not None:
                self.analyze_text(r.heredoc_body, depth + 1)
            if r.op == "<<<" and r.target is not None:
                self.analyze_text(r.target.text, depth + 1)
            if r.op == "<" and self.dotenv(r.target):
                raise Blocked("runs a .env file as a shell script", ADVICE_DOTENV)

    def interpreter(self, a0: str, words: list, cmd: Cmd, depth: int) -> None:
        code_words = []
        stdin_code = False
        implied_print = False
        i = 1
        py = bool(_PYTHONS.match(a0))
        while i < len(words):
            t = words[i].text
            if py and t.startswith("-") and not t.startswith("--") and "c" in t[1:]:
                if i + 1 < len(words):
                    code_words.append(words[i + 1])
                break
            if not py and t in ("-e", "-E", "--eval", "-p", "--print", "-r", "-pe", "-ne", "-le"):
                if t in ("-p", "--print") and a0 in ("node", "deno", "bun"):
                    implied_print = True  # node -p prints the expression's value
                if i + 1 < len(words):
                    code_words.append(words[i + 1])
                i += 2
                continue
            if t == "-":
                stdin_code = True
                break
            if t == "-m" and py:
                module = words[i + 1].text if i + 1 < len(words) else ""
                if module.split(".")[0] == "dotenv":
                    raise Blocked("runs `python -m dotenv`, whose `list` and `get` print .env values",
                                  ADVICE_DOTENV)
                if module == "pytest" and self.sources_dotenv:
                    raise Blocked("runs pytest in a command that sourced .env", ADVICE_PYTEST)
                return
            if t.startswith("-"):
                i += 1
                continue
            if self.dotenv(words[i]):
                raise Blocked("runs a .env file through an interpreter", ADVICE_DOTENV)
            return  # a script file: its contents are not visible here
        else:
            stdin_code = True
        bodies = [w.text for w in code_words]
        for w in code_words:
            if self.bad_refs(w):
                raise Blocked("expands a secret variable into inline code", ADVICE_ECHO)
        if stdin_code:
            for r in cmd.redirs:
                if r.op in ("<<", "<<-") and r.heredoc_body is not None:
                    bodies.append(r.heredoc_body)
                if r.op == "<<<" and r.target is not None:
                    bodies.append(r.target.text)
        for code in bodies:
            self.inline_code(code, a0, implied_print, depth)

    def inline_code(self, code: str, lang: str, implied_print: bool, depth: int) -> None:
        if _CODE_DOTENV.search(code):
            raise Blocked("opens a secrets file (.env or var/railway/*key*.txt) from inline code", ADVICE_DOTENV)
        if _CODE_SHELLOUT.search(code):
            self.inline_shell_commands(code, depth)
        stripped = code
        for rx in _CODE_SAFE_FORMS:
            stripped = rx.sub(" ", stripped)
        prints = implied_print or bool(_CODE_OUTPUT.search(stripped))
        if lang == "ruby" and re.search(r"(?<![\w.])pp?\b", stripped):
            prints = True
        if not prints:
            return
        m = _CODE_SECRET.search(stripped) if _CODE_ENV_ACCESS.search(stripped) else None
        if m:
            raise Blocked(f"prints a secret ({m.group(0)}) from inline code", ADVICE_INLINE)
        if _CODE_WHOLE_ENV.search(stripped):
            raise Blocked("prints the whole environment or settings from inline code", ADVICE_INLINE)
        for m in _CODE_SETTINGS_ALIAS.finditer(stripped):
            n = re.escape(m.group(1))
            whole = (
                rf"\bprint\s*\(\s*{n}\s*[,)]|\b{n}\s*\.\s*(?:__dict__|model_dump)|"
                rf"\b(?:vars|repr|str|dict)\s*\(\s*{n}\s*\)|\{{{n}(?:!r|!s)?\}}"
            )
            if re.search(whole, stripped):
                raise Blocked("prints the whole settings object from inline code", ADVICE_INLINE)

    def inline_shell_commands(self, code: str, depth: int) -> None:
        """Inline code that shells out (`os.system("env")`, `subprocess.run(["printenv"])`):
        check each string literal, and each list of literals joined, as a shell command."""
        commands = []
        for m in _CODE_LIST.finditer(code):
            parts = [a or b for a, b in _CODE_STRING.findall(m.group(1))]
            if len(parts) > 1:
                commands.append(" ".join(parts))
        commands.extend(a or b for a, b in _CODE_STRING.findall(code))
        for text in commands[:64]:
            if not text.strip() or len(text) > 4000:
                continue
            sub = Analyzer(self.cwd)
            try:
                sub.analyze_text(text, depth + 1)
            except Blocked as b:
                raise Blocked(f"shells out from inline code to a command that {b.reason}",
                              b.advice, b.rule) from None

    # --- rules
    def check_dotenv(self, a0, eff, args, arg_refs, in_redirs) -> None:
        if a0 in (".", "source") and not in_redirs:
            if all(self.secret_kind(w) == "env" for w in arg_refs):
                self.sources_dotenv = True
                return
            raise Blocked("sources a key file, whose error message would print the key", ADVICE_DOTENV)
        if a0 == "" and in_redirs:
            raise Blocked("reads a secrets file (`$(< file)` / a bare `< file`)", ADVICE_DOTENV)
        if a0 in _GREPS and grep_prints_no_lines(args):
            return
        if a0 == "wc" or (a0 in _READERS_OK_ON_DOTENV and not in_redirs):
            return
        if a0 == "git" and args and args[0] in ("check-ignore", "status", "ls-files"):
            return
        if a0 in ("cp", "mv", "install", "ln") and not in_redirs:
            if all(w is eff[-1] for w in arg_refs):
                return  # .env is only the destination: writing, not reading
        if a0 == "find" and not in_redirs and not any(
            a in ("-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprintf", "-fls") for a in args
        ):
            return
        what = f"`{a0}`" if a0 else "a redirect"
        raise Blocked(f"reads a secrets file (.env*, var/railway/*key*.txt or a credential file) "
                      f"with {what}", ADVICE_DOTENV)

    def root_holds_dotenv(self, root: str) -> bool:
        text = self.expand(root)
        if text.startswith(("~", "$")):
            return True
        if os.path.isabs(text):
            return _tree_holds_dotenv(text)
        if not self.curdir or not os.path.isdir(self.curdir):
            # no working directory to look in: the places a .env usually sits
            return text.rstrip("/") in ("", ".", "..", "*") or text.startswith("../")
        if text.rstrip("/") == "*":
            text = "."
        return _tree_holds_dotenv(os.path.join(self.curdir, text))

    def check_recursive_grep(self, a0: str, args: list) -> None:
        """`grep -rn OPENROUTER .` reads ./.env too (grep ignores .gitignore) and prints every
        line of it that matches, whatever the pattern. Refused when a search root holds a .env
        and the options do not exclude it. `rg` skips gitignored and hidden files by default,
        so only its --no-ignore/--hidden/-u forms count."""
        if a0 not in _GREPS or grep_prints_no_lines(args):
            return
        letters = short_flags(args)
        longs = {a.split("=", 1)[0] for a in args if a.startswith("--")}
        if a0 == "rg":
            recursive = bool(longs & {"--no-ignore", "--hidden", "--unrestricted"}) or "u" in letters
        else:
            recursive = bool(letters & {"r", "R"}) or bool(longs & {"--recursive", "--dereference-recursive"})
        if not recursive:
            return
        excludes = [a for a in args if a.startswith(("--exclude", "--glob", "-g"))]
        positional, patterns, i = [], [], 0
        while i < len(args):
            a = args[i]
            if a in ("-e", "--regexp", "-f", "--file") and i + 1 < len(args):
                patterns.append(args[i + 1])
                i += 2
                continue
            if a in ("--exclude", "--include", "--glob", "--iglob", "-g", "--type", "-t", "-A", "-B",
                     "-C", "-m", "--max-count", "--exclude-dir") and i + 1 < len(args):
                i += 2
                continue
            if a.startswith("-") and a != "-":
                i += 1
                continue
            positional.append(a)
            i += 1
        if not patterns and positional:
            patterns.append(positional.pop(0))
        roots = positional or ["."]
        # A match-everything pattern over a tree holding var/railway prints the raw key files.
        match_all = any(p in ("", ".", "^", "$", ".*", ".+", "\\w", "\\S", "[[:alnum:]]") for p in patterns)
        reaches_var = [
            r for r in roots
            if r.rstrip("/") in ("", ".", "..", "~", "*", "var") or r.startswith(("/", "~", "$", "../"))
        ]
        if match_all and reaches_var and not any("railway" in a for a in excludes):
            raise Blocked(
                "greps everything under a tree that holds var/railway key files, which prints them",
                "name the directories you mean, or add `--exclude-dir=railway`.",
            )
        if _dotenv_excluded(args, a0):
            return
        if any(self.root_holds_dotenv(r) for r in roots):
            raise Blocked(
                "greps recursively over a directory that holds a .env file, which prints every "
                "line of it the pattern matches",
                ADVICE_GREP_TREE,
            )

    def check_env_dumps(self, a0: str, args: list, eff: list) -> None:
        if a0 == "printenv":
            raise Blocked("runs `printenv`, which prints variable values", ADVICE_ENVDUMP)
        if a0 == "export" and (not args or "-p" in args):
            raise Blocked("runs `export` with no names, which prints every exported value", ADVICE_ENVDUMP)
        if a0 in ("declare", "typeset", "readonly", "local"):
            opts = [a for a in args if a[:1] in "-+"]
            names = [a for a in args if a[:1] not in "-+"]
            letters = set("".join(o[1:] for o in opts))
            if "p" in letters:
                raise Blocked(f"runs `{a0} -p`, which prints variable values", ADVICE_ENVDUMP)
            if "m" in letters:
                raise Blocked(f"runs `{a0} -m`, which prints every matching variable's value", ADVICE_ENVDUMP)
            if not names and not letters & set("fF"):
                raise Blocked(f"runs `{a0}` with no names, which prints variable values", ADVICE_ENVDUMP)
        if a0 == "set":
            if not args:
                raise Blocked("runs bare `set`, which prints every variable", ADVICE_ENVDUMP)
            letters = short_flags([a for a in args if a.startswith("-")])
            if letters & set("xv") or any(
                args[i] == "-o" and i + 1 < len(args) and args[i + 1] in ("xtrace", "verbose")
                for i in range(len(args))
            ):
                self.xtrace = True
        if a0 == "ps":
            for a in args:
                if (not a.startswith("-") and a.isalpha() and "e" in a) or (
                    a.startswith("-") and not a.startswith("--") and "E" in a
                ):
                    raise Blocked("runs `ps` with environment display", ADVICE_ENVDUMP)
        if a0 in _AWKS and any("ENVIRON" in a for a in args if not a.startswith("-")):
            raise Blocked("reads ENVIRON in awk, which prints variable values", ADVICE_ENVDUMP)
        if a0 in _JQS and any(re.search(r"(?<![.\w$])env\b|\$ENV\b", a) for a in args if not a.startswith("-")):
            raise Blocked("reads the environment in jq (`env` / `$ENV`)", ADVICE_ENVDUMP)
        if a0 == "gh" and args[:2] == ["auth", "token"]:
            raise Blocked("prints the GitHub token", ADVICE_LITERAL)
        if a0 == "gh" and args[:2] == ["auth", "status"] and ("-t" in args or "--show-token" in args):
            raise Blocked("prints the GitHub token", ADVICE_LITERAL)
        if a0 == "aws" and args[:2] == ["configure", "export-credentials"]:
            raise Blocked("prints AWS credentials", ADVICE_LITERAL)
        if a0 == "aws" and args[:2] == ["configure", "get"] and any(SECRET_NAME.search(a) for a in args[2:]):
            raise Blocked("prints an AWS secret", ADVICE_LITERAL)
        if a0 == "security" and args and args[0] in ("find-generic-password", "find-internet-password") \
                and ({"w", "g"} & short_flags(args[1:])):
            raise Blocked("prints a keychain password", ADVICE_LITERAL)
        if a0 == "docker" and (args[:1] == ["inspect"] or args[:2] == ["container", "inspect"]):
            fmt = None
            for i, a in enumerate(args):
                if a in ("-f", "--format") and i + 1 < len(args):
                    fmt = args[i + 1]
                elif a.startswith("--format="):
                    fmt = a.split("=", 1)[1]
            if fmt is None or re.search(r"Env|\.Config\s*\}\}|json\s+\.\s*\}\}|\{\{\s*\.\s*\}\}", fmt):
                raise Blocked("runs `docker inspect`, which prints the container's environment",
                              "pass a `--format` that names only the fields you need.")

    def check_git(self, a0: str, args: list) -> None:
        if a0 != "git":
            return
        i, gdir = 0, None
        while i < len(args):
            t = args[i]
            if t == "-C" and i + 1 < len(args):
                gdir = args[i + 1]
                i += 2
                continue
            if t in ("-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env"):
                i += 2
                continue
            if t.startswith("-"):
                i += 1
                continue
            break
        sub = args[i] if i < len(args) else ""
        rest = args[i + 1:]
        if sub == "push" and self.worktree:
            raise self.live("runs `git push`")
        if sub == "push":
            base = self.curdir or os.getcwd()
            path = os.path.expanduser(self.expand(gdir)) if gdir else ""
            self.pushes.append(os.path.normpath(os.path.join(base, path)) if path else base)
        if sub == "credential" and rest[:1] == ["fill"]:
            raise Blocked("runs `git credential fill`, which prints a stored credential", ADVICE_LITERAL)
        if sub == "commit":
            base = self.curdir or os.getcwd()
            path = os.path.expanduser(self.expand(gdir)) if gdir else ""
            self.commits.append(os.path.normpath(os.path.join(base, path)) if path else base)

    def check_worktree(self, a0: str, eff: list) -> None:
        texts = [self.expand(w.text) for w in eff]
        if not texts:
            return
        if a0 == "gh" and texts[1:3] == ["pr", "merge"]:
            raise self.live("merges a pull request (a deploy when it lands on main)")
        if _VAR_OPS.search(texts[0]):
            raise self.live("runs a var/ops script (the orchestrator's live-system scripts)")
        if a0 in _VIEWERS:
            return
        if any(_VAR_OPS.search(t) for t in texts[1:]):
            raise self.live("runs or copies a var/ops script (the orchestrator's live-system scripts)")
        if any(_LIVE_HOST.search(t) for t in texts[1:]):
            raise self.live("reaches the live database host")

    def check_locked(self, a0: str, eff: list, cmd: Cmd) -> None:
        texts = [self.expand(w.text) for w in eff[1:]] + [
            self.expand(r.target.text) for r in cmd.redirs if r.target is not None
        ]
        if not any(_LOCKED.search(t) for t in texts) or a0 in _LOCKED_LISTERS:
            return
        if a0 == "git" and texts[:1] and texts[0] in _LOCKED_GIT_OK and not (
            set(texts) & _LOCKED_GIT_READERS
        ):
            return
        raise Blocked("reads the locked exam answers (evals/locked/) from an agent worktree",
                      ADVICE_LOCKED, RULE_LOCKED)

    def check_railway(self, a0: str, args: list, cmd: Cmd, pipe: list, idx: int) -> None:
        if a0 != "railway" or not args or args[0] not in ("variables", "variable", "vars"):
            return
        kv = "--kv" in args or "-k" in args or bool(short_flags(args) & {"k"})
        to_file = any(r.op in (">", ">>", ">|", "&>", "&>>") and (r.target is None or r.target.text != "/dev/null")
                      for r in cmd.redirs)
        names_only = False
        for later in pipe[idx + 1:]:
            if not later.words:
                break
            name = cmdname(later.words[0].text)
            largs = [w.text for w in later.words[1:]]
            if name == "tee":
                break
            if name == "cut" and cut_takes_names(largs):
                names_only = True
                break
        if kv and names_only and not to_file:
            return
        raise Blocked("runs `railway variables`, which prints deployment values", ADVICE_RAILWAY)

    def check_echo(self, a0: str, eff: list, args: list, cmd: Cmd) -> None:
        if a0 == "printf" and "-v" in args:
            # `printf -v x ... "$SECRET"` prints nothing, but x now holds the secret
            j = args.index("-v")
            if j + 1 < len(args) and any(self.bad_refs(w) for w in eff[1:]):
                self.tainted.add(args[j + 1])
        elif a0 in ("echo", "printf", "print"):
            for w in eff[1:]:
                bad = self.bad_refs(w)
                if bad:
                    raise Blocked(f"prints the secret variable {bad[0]} with `{a0}`", ADVICE_ECHO)
        quiet_grep = a0 in _GREPS and grep_prints_no_lines(args)
        for r in cmd.redirs:
            if r.op == "<<<" and r.target is not None and self.bad_refs(r.target):
                if not (quiet_grep or a0 in ("wc", "test", "[") or a0 in _SHELLS or _PYTHONS.match(a0 or "")):
                    raise Blocked("prints a secret variable through a here-string", ADVICE_ECHO)
            body = r.expanded
            if body is not None and a0 in _PRINTERS | {"echo", "printf", ""} and self.bad_refs(body):
                raise Blocked("prints a secret variable through a heredoc", ADVICE_ECHO)
        if a0 in _HTTP:
            verbose = bool(short_flags(args) & {"v"}) or any(
                a in ("--verbose", "--trace", "--trace-ascii", "--debug", "-d") for a in args
            )
            if verbose and any(self.bad_refs(w) for w in eff[1:]):
                raise Blocked(f"sends a secret with verbose `{a0}`, which echoes the headers", ADVICE_ECHO)


def cut_takes_names(args: list) -> bool:
    delim, fields = None, None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-d", "--delimiter") and i + 1 < len(args):
            delim, i = args[i + 1], i + 2
            continue
        if a.startswith("--delimiter="):
            delim = a.split("=", 1)[1]
        elif a.startswith("-d"):
            delim = a[2:]
        elif a in ("-f", "--fields") and i + 1 < len(args):
            fields, i = args[i + 1], i + 2
            continue
        elif a.startswith("--fields="):
            fields = a.split("=", 1)[1]
        elif a.startswith("-f"):
            fields = a[2:]
        i += 1
    return delim == "=" and fields == "1"


# --------------------------------------------------------------------------- git commit scan


def _hook_repo_config() -> str | None:
    """The `.gitleaks.toml` of the checkout this hook file lives in (`<repo>/.claude/hooks/`).
    The hook's own copy, not the committing branch's, so a branch cannot weaken its own scan."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(os.path.dirname(here)), ".gitleaks.toml")
    return path if os.path.isfile(path) else None


def _git(args: list, cwd: str, env=None, timeout: float = 30):
    import subprocess

    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def scan_commit(repo_dir: str) -> list:
    """gitleaks findings ("rule file:line") in every change a commit from repo_dir could take in:
    staged, unstaged and untracked-not-ignored files, gathered into a throwaway copy of the
    index so the real one is untouched. [] when clean, or when the scan cannot run."""
    import shutil  # imported here so only a commit pays their start-up cost
    import subprocess
    import tempfile

    gitleaks = shutil.which("gitleaks")
    if gitleaks is None or not os.path.isdir(repo_dir):
        return []
    try:
        top = _git(["rev-parse", "--show-toplevel"], repo_dir)
        if top.returncode != 0:
            return []
        top_dir = top.stdout.strip()
        idx = _git(["rev-parse", "--git-path", "index"], top_dir)
        if idx.returncode != 0:
            return []
        index = os.path.join(top_dir, idx.stdout.strip())
        with tempfile.TemporaryDirectory(prefix="no-secrets-") as td:
            tmp_index = os.path.join(td, "index")
            if os.path.isfile(index):
                shutil.copyfile(index, tmp_index)
            env = dict(os.environ, GIT_INDEX_FILE=tmp_index)
            if _git(["add", "-A"], top_dir, env=env, timeout=60).returncode != 0:
                return []
            report = os.path.join(td, "report.json")
            cmd = [gitleaks, "git", "--staged", "--redact", "--no-banner", "--exit-code", "1",
                   "--report-format", "json", "--report-path", report]
            config = _hook_repo_config() or (
                os.path.join(top_dir, ".gitleaks.toml")
                if os.path.isfile(os.path.join(top_dir, ".gitleaks.toml")) else None
            )
            if config:
                cmd += ["--config", config]
            cmd.append(top_dir)
            proc = subprocess.run(cmd, cwd=top_dir, env=env, capture_output=True, text=True, timeout=90)
            if proc.returncode != 1:
                return []
            with open(report, encoding="utf-8") as fh:
                findings = json.load(fh)
    except Exception:
        return []
    return [f"{f.get('RuleID')} {f.get('File')}:{f.get('StartLine')}" for f in findings]


# --------------------------------------------------------------------------- entry points

_RISKY_TEXT = re.compile(
    r"\.env|var/railway|var/ops|printenv|environ|dotenv|railway|rlwy\.net|/proc/|credential|"
    r"\b(?:env|set|export|declare|typeset)\b|\bpush\b|evals/locked|" + SECRET_NAME.pattern,
    re.I,
)


def _push_problems(repo_dir: str) -> list:
    """Reasons the checkout at repo_dir may not push (`push_gate.py`, next to this file)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import push_gate
    finally:
        sys.path.pop(0)
    return push_gate.problems(repo_dir)


def _with_budget(seconds: float, fn, *args):
    """Run fn under a wall-clock budget (SIGALRM); raises OverBudget when it runs out."""
    if not hasattr(signal, "setitimer"):
        return fn(*args)

    def on_alarm(signum, frame):
        raise OverBudget()

    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def analyze(command: str, cwd: str | None = None):
    """(verdict, analyzer): verdict is None when the command may run, else
    (reason, advice, rule)."""
    for rx, what in KEY_LITERALS:
        if rx.search(command):
            return (f"contains a literal shaped like {what}", ADVICE_LITERAL, RULE_PRINT), None
    an = Analyzer(cwd)
    try:
        an.analyze_text(command, 0)
        if an.xtrace and an.sources_dotenv:
            raise Blocked("turns on xtrace (`set -x` / `bash -x`) while sourcing .env, which "
                          "prints every assignment", ADVICE_DOTENV)
    except Blocked as b:
        return (b.reason, b.advice, b.rule), an
    return None, an


def check(command: str, cwd: str | None = None):
    """None when the command may run, else (reason, advice)."""
    verdict, _ = analyze(command, cwd)
    return None if verdict is None else verdict[:2]


def check_path_tool(tool: str, tool_input: dict, cwd: str | None = None):
    if tool == "Read":
        fields = [("file_path", tool_input.get("file_path"))]
    else:
        fields = [("path", tool_input.get("path")), ("glob", tool_input.get("glob"))]
    worktree = bool(cwd) and bool(_WORKTREE.search(cwd.replace("\\", "/")))
    for name, value in fields:
        if not isinstance(value, str) or not value.strip():
            continue
        text = value.replace("\\", "/")
        verb = "reads" if tool == "Read" else f"searches (its `{name}`)"
        if is_secret_path(text):
            return (f"{verb} a secrets file (.env*, var/railway/*key*.txt or a credential file)",
                    ADVICE_DOTENV, RULE_PRINT)
        if worktree and _LOCKED.search(text):
            return (f"{verb} the locked exam answers (evals/locked/) from an agent worktree",
                    ADVICE_LOCKED, RULE_LOCKED)
    return None


def decide(tool: str, tool_input: dict, cwd: str | None):
    if tool in ("Bash", "Monitor"):
        command = tool_input.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        verdict, an = _with_budget(ANALYSIS_BUDGET_SECONDS, analyze, command, cwd)
        if verdict is None and an is not None:
            for repo_dir in dict.fromkeys(an.commits):
                findings = scan_commit(repo_dir)
                if findings:
                    shown = ", ".join(findings[:5]) + (f" and {len(findings) - 5} more" if len(findings) > 5 else "")
                    return (f"would commit what gitleaks flags as a secret ({shown})",
                            ADVICE_COMMIT, RULE_COMMIT)
            for repo_dir in dict.fromkeys(an.pushes):
                reasons = _push_problems(repo_dir)
                if reasons:
                    return ("would push, and " + "; ".join(reasons), ADVICE_PUSH, RULE_PUSH)
        return verdict
    if tool in ("Read", "Grep"):
        return check_path_tool(tool, tool_input, cwd)
    return None


def unchecked_verdict(tool_input: dict):
    """After an internal error or a blown time budget: block only if the raw text looks risky."""
    text = " ".join(v for v in tool_input.values() if isinstance(v, str))
    m = _RISKY_TEXT.search(text)
    if not m:
        return None
    return (f"could not be checked (the hook hit an internal error or its time limit) and it "
            f"mentions `{m.group(0)}`", ADVICE_UNCHECKED, RULE_PRINT)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        return 0  # a payload that cannot be parsed: fail open
    if not isinstance(payload, dict):
        return 0
    tool = payload.get("tool_name", "Bash")
    tool_input = payload.get("tool_input")
    if not isinstance(tool, str) or not isinstance(tool_input, dict):
        return 0
    cwd = payload.get("cwd")
    cwd = cwd if isinstance(cwd, str) and cwd else None
    try:
        verdict = decide(tool, tool_input, cwd)
    except Exception:  # a bug, a RecursionError, or OverBudget
        verdict = unchecked_verdict(tool_input)
    if verdict is None:
        return 0
    reason, advice, rule = verdict
    what = "command" if tool in ("Bash", "Monitor") else f"{tool} call"
    sys.stderr.write(
        f"no-secrets hook blocked this {what}: it {reason}.\n"
        f"Instead: {advice}\n"
        f"(.claude/hooks/no-secrets.py; CLAUDE.md rule '{rule}'.)\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
