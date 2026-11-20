# `portolan`: the Portolan CLI

Use and test a running Portolan app from the shell, no browser. **Every agent tests app changes with this, against a local app.**

```bash
uv run portolan <command> [args] [--json] [--session file]     # or: bin/portolan <command> ...
```

- **Targets.** `--target local` (the default) talks to an app you started yourself at `--base-url` (default `http://127.0.0.1:8000`, or `$PORTOLAN_BASE_URL`). It starts nothing, and it refuses any host that is not loopback. `--target https://host` talks to a deployed app. Agents never point it at the live site; the orchestrator does.
- **It goes through the front door.** It signs in with the org key and uses the same pages and routes the browser does (`/atlas`, `/notes/<slug>`, `/originals/...`, `/chat/<id>/stream`, `/review/...`). Only `search --k/--since` and `mcp ...` use the MCP endpoint, with the MCP key. The shared client lives in `portolan/client/`; `scripts/verify.py` uses the same one.
- **Keys** are read from files and never printed, logged, or written to the session file. Org key: `--org-key-file`, else (local target) `$PORTOLAN_ORG_KEY`, else `var/railway/org-key.txt`. MCP key: `--mcp-key-file`, else `$PORTOLAN_MCP_ORG_KEY`, else `var/railway/mcp-key.txt`. A URL target reads the files first, as `verify.py` does.
- **Sessions.** `portolan new` writes `.portolan-session.json` in the current directory (override with `--session <file>` or `$PORTOLAN_SESSION`). It holds the target, your name, the chat conversation id and a log of commands, so a chat can carry on across separate shell calls. Every call signs in again; no cookie is stored. The file is gitignored.
- **Output** is plain text by default; `--json` gives the same data as JSON everywhere (on failure: `{"exit": N, "message": "..."}`).
- **Exit codes:** `0` ok, `1` assertion or validation failure, `2` usage error, `3` the app refused (a 4xx, or a key it does not accept), `4` the app errored or timed out.
- **No paid calls of its own.** `ask` is one chat question, which is a paid model call on any app that has a real model. Against the seeded test app (below) the model is a stub.

## Run a local app to test against

```bash
createdb portolan_test_cli        # once; any name starting portolan_test
uv run python -m tests.verify_seed --database-url postgresql://localhost/portolan_test_cli \
    --port 8765 --workdir /tmp/portolan-cli --chat-stub
# in another shell:
uv run portolan new --base-url http://127.0.0.1:8765 --org-key-file /tmp/portolan-cli/org-key.txt
```

The seed is the one `tests/test_verify_script.py` uses (the fake extractor over `corpus/out`, two review rows, a weekly report). `--chat-stub` answers chat from the notes with no model call. For your own dev database use `python -m portolan.web` and pass its keys with `--org-key-file`.

## Commands

| Command | What it does |
|---|---|
| `new [--target T] [--base-url U] [--editor NAME]` | Start (or replace) the session file. |
| `ask "<question>" [--new-conversation]` | Chat through the real stream. Prints the answer, then each citation (note, claim, original link, quote, and whether the quote was found), withheld count, outcome, latency. Cost is not exposed by the app, so it prints `not exposed`. Every citation is followed the way a reader clicks it: note page, claim, original, quote. |
| `search "<q>" [--k N] [--since YYYY-MM-DD]` | The Atlas search the browser uses. With `--k` or `--since` it calls the MCP `search` tool instead (the page has neither). |
| `note <slug>` | Title, type, dates, claims (with quote and original link), links, backlinks. |
| `open <original link> [--quote TEXT]` | Fetch an `/originals/...` page; check the cited block is there and, with `--quote`, that the text is on it (the production quote gate's rule). Exit 1 if not. |
| `accounts [filter]` / `account <name or slug>` | The account list; one account's note plus its open actions. `account` exits 1 if the name matches none or several. |
| `actions [--owner NAME]` | Open actions grouped by owner, with ids, due dates and overdue flags. |
| `review list [--kind K] [--page N]` | Open review rows. |
| `review decide <id> accept\|reject --yes [--reason TEXT]` | Confirm (`accept`) or dismiss (`reject`) a row. **Writes**; without `--yes` it exits 2. Attributed to `--editor`. |
| `report list` / `report show <slug>` | The weekly reports the app holds; one as markdown. |
| `report weekly ...` / `report monday ...` | Runs `python -m portolan.report weekly\|monday` in this process's environment. Local target only. Put global flags before `report`; everything after it goes to the module (use `--` before `--help`). It reaches whatever database and model that environment names. |
| `usage` | The `/admin/usage` numbers: totals, weeks, people, unanswered questions. |
| `health` | `/healthz`, latency, and whether the key signs in. Exit 3 if the key is refused, 4 if the app is down. |
| `mcp tools` / `mcp call <tool> '<json>'` | List the MCP tools; call one with JSON arguments. |
| `script <file.json> [--keep-session]` | Run a scenario (below). |

## Example session

```
$ uv run portolan new --base-url http://127.0.0.1:8793
session .portolan-session.json: target local (http://127.0.0.1:8793), editor portolan-cli

$ uv run portolan ask "Which accounts have asked for Tidewater Link to work with Epic?"
Q: Which accounts have asked for Tidewater Link to work with Epic?
A: [[priya-natarajan]] promised Wrenfield Health a Tidewater Link/Epic integration test plan by 2026-09-10. (owner: Priya Natarajan) Wrenfield Health told Halvard on 2026-08-10 that it would defer any Tidewater Link/Epic integration discussion until 2027, effectively asking to shelve the request. ([[tidewater-link]] [[wrenfield-health]])

outcome answered | citations 2 (0 broken) | withheld 0 | 267.0 ms | cost not exposed | conversation 4
  1. action-4-priya-natarajan-promised-wrenfield-health-a-tidewater-link-e claim 36
     original: /originals/S-call-020?at=09%3A15%3A16&from=chat&m=12
  2. signal-request-wrenfield-health claim 27
     original: /originals/S-call-016?at=09%3A02%3A55&from=chat&m=12

$ uv run portolan note signal-request-wrenfield-health
Request — Wrenfield Health  [signal, living]  signal-request-wrenfield-health
...
Claims (2):
  - Wrenfield Health told Halvard on 2026-08-10 that it would defer any Tidewater Link/Epic integration discussion until 2027, ...
    S-call-016 @ 09:02:55  /originals/S-call-016?at=09%3A02%3A55
Links (2): wrenfield-health, tidewater-link
Backlinks (2): report-2026-W36 (body), wrenfield-health (body)

$ uv run portolan open "/originals/S-call-016?at=09%3A02%3A55" --quote "Epic"
/originals/S-call-016?at=09%3A02%3A55: 200, document with 74 block(s)
locator 09:02:55: found
quote "Epic": exact
```

## Test: `script <file.json>`

Runs a scenario and **exits 1 on the first failed expectation** (or with the command's own code, 2, 3 or 4, when a step that should have worked did not). Examples are in `tests/cli_scripts/` and run in CI through `tests/test_cli.py`.

```json
{ "name": "Epic requests",
  "steps": [
    { "do": "ask \"Which accounts asked for Tidewater Link to work with Epic?\"",
      "expect": { "outcome": "answered", "citations": ">=2", "cited_notes": "contains signal-" } },
    { "do": "note no-such-note", "exit": 3 },
    { "do": "open \"/originals/S-call-016?at=09%3A02%3A55\" --quote \"Epic\"",
      "expect": ["ok==true", "quote_match==exact"] }
  ] }
```

- A script starts a fresh session first (use `--keep-session` to continue the current one). Put `--base-url`, `--org-key-file` and the other global flags on the `script` command; steps take only their own arguments.
- A step has `do` (any command, without `portolan`), `expect`, and `exit` (the exit code it must end with, default 0). A step with only `expect` checks the previous result again.
- `expect` is an object (`{"citations": ">=2"}`), a list (`["citations>=2"]`) or one string. Operators: `>= <= > < == != contains`. A plain value means `==`. `contains` is case-insensitive and, on a list, matches if any item contains the text. On a list or object, a number compares its size, so `"citations": ">=2"` counts citations. Dotted paths reach inside (`claims.0.quote`). `exit` is always a metric.
- Metrics are the keys of each command's `--json` output:

| Command | Keys worth asserting |
|---|---|
| `ask` | `outcome` (`answered`, `nothing_recorded`, `no_answer`, `index_broken`, `error`), `answer`, `citations` (list), `cited_notes`, `broken_citations`, `withheld`, `latency_ms`, `conversation_id`, `new_conversation` |
| `search` | `count`, `slugs`, `top`, `hits`, `control_ok`, `source` |
| `note`, `account` | `title`, `type`, `status`, `updated`, `claims`, `links`, `backlinks`, `open_actions` |
| `open` | `ok`, `block_found`, `quote_match`, `quote_in_cited_block`, `blocks`, `problems` |
| `accounts` | `total`, `accounts` |
| `actions` | `open`, `overdue`, `owners`, `actions`, `closed_this_week` |
| `review list`, `review decide` | `open`, `cards`, `pages`; `status`, `notice`, `already_handled` |
| `usage` | `totals.<name>` (`questions`, `active_users`, ...), `weeks`, `people`, `unanswered` |
| `health` | `status`, `ok`, `latency_ms`, `auth` |
| `mcp tools`, `mcp call` | `count`, `names`; `is_error`, `result` |

`tests/cli_scripts/smoke.json` (every read path), `refusals.json` (the exit codes) and `demo_live.json` are the shipped examples. **`demo_live.json` is the ten questions in `scripts/demo_questions.txt` in one conversation, for the orchestrator to run against the live site** (`uv run portolan script tests/cli_scripts/demo_live.json --target <live url>`, with the org key in `var/railway/org-key.txt`). It is never part of CI, and `tests/test_cli.py` only checks that it still asks the ten questions in that file.

## Related tools

- `scripts/verify.py` proves a whole running app page by page and writes the result to `var/proof/`; `portolan` is for a human or agent poking one thing. Both use `portolan/client/`.
- `python -m tests.verify_seed` is the seeded local app.
