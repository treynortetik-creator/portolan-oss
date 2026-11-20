# Portolan MCP server

Wraps Portolan's search, notes, sources, coverage ledger and standing
questions as MCP tools (portolan-spec-v2.md sec 10: "MCP, second... Runs
on the customer's own AI; costs them nothing from us"). One factory,
`create_server(settings)`, builds a server usable over either transport:

- **stdio** — a local client (Claude Code, Claude Desktop) spawns this
  process directly. No key needed.
- **streamable HTTP** — a hosted deployment, gated by a shared org key.

SDK: [`mcp`](https://github.com/modelcontextprotocol/python-sdk) 2.2.0
(the official Model Context Protocol Python SDK). Note for anyone reading
older MCP examples: this version renamed `FastMCP` to
`mcp.server.mcpserver.MCPServer` — `server.tool()`, `server.run(...)`,
`server.streamable_http_app(...)` below are that class's API.

## Tools

| Tool | Does |
|---|---|
| `search(query, k=12, expand=0, types=None, since=None)` | Fused keyword + vector search over notes, with recency boost, `control_ok` diagnosis, and `expand` to pull in linked notes. `since` and recency read each note's `content_date` (newest claim recorded date, else last write). |
| `open_note(slug, claims_limit=50)` | A note's frontmatter, body, links, and the first page of its claims (source + locator + quote + both clocks), with `claims_page.next_cursor` for the rest. |
| `list_claims(slug=None, cursor=None, limit=50)` | The next page of a note's claims (pass `cursor`), or a new listing (pass `slug`), over a frozen snapshot. |
| `open_source(slug)` | A source's fields, extraction status, the notes extracted from it, and `original_url`. |
| `add_standing_question(reader, question)` | Registers a standing question the weekly report should keep answering. |
| `list_standing_questions(reader=None)` | Lists standing questions, optionally by reader. |
| `coverage(since_days=7)` | The coverage ledger for the last N days: seen / filtered / extracted / partial / failed / sampled, per connector per day. |
| `graph_neighbors(slug, depth=1)` | Notes linked to or from a note, out to N hops. |
| `precedents(theme_slug, account_slug=None)` | Commitments and decisions that followed a theme before, and how long resolution took. |

One resource: `portolan://ai-guide` (`AI_GUIDE.md` in this package), the
guide for the consuming model.

Every call writes a `usage_events` row (`surface='mcp'`, migration
`0003_standing_questions.sql` extends the surface check constraint to
allow it) logging the tool name, argument sizes, and a hit count — never a
note body, claim text, or query string.

## The door is a hostile surface (checklist item 26, 2026-09-22)

Both transports share four controls; `docs/mcp-hosted.md` has the full
operator's view.

- **AI-access switch** (`access.py`, migration 0031): per tenant, checked
  before every tool call and every list/read. A new database is closed
  until `python -m portolan.mcp_server access on`; a database that existed
  before 0031 was grandfathered open. A closed door is JSON-RPC error
  `-32010`, never an empty result.
- **Sanitized output** (`sanitize.py`): every string leaving the door has
  invisible characters stripped, is NFKC-normalised, and is wrapped in
  `<<UNTRUSTED EVIDENCE ref=...>>` markers when it reads like an
  instruction; dict results carry `_portolan_security` with the SHA-256
  of each altered stored string. `tools.py` stays raw on purpose: the
  action researcher calls it directly and its evidence gate is byte-equal.
- **Cursors** (`cursors.py`, `tools.claims_page`): signed keyset cursors
  over a frozen snapshot, so pages never move under a re-crawl.
- **AI guide** (`AI_GUIDE.md`), served as the resource above.

## Local use: stdio

```bash
python -m portolan.mcp_server --stdio
```

Point a client at it. Claude Code (`.mcp.json` in the project, or
`claude mcp add`) and Claude Desktop (`claude_desktop_config.json`) both
take the same shape:

```json
{
  "mcpServers": {
    "portolan": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/living-knowledge-graph",
                "python", "-m", "portolan.mcp_server", "--stdio"],
      "env": {
        "DATABASE_URL": "postgresql://localhost/portolan"
      }
    }
  }
}
```

(Swap `"command": "uv", "args": [...]` for a bare `python3` invocation with
the venv already active if you'd rather not shell out through `uv`.)

## Hosted use: streamable HTTP

**On the deployed web service, this is already wired up for you** --
`portolan.web.app.create_app` mounts this transport at `/mcp` on the same
process, gated by `PORTOLAN_MCP_ORG_KEY`, whenever that env var is set;
see `docs/mcp-hosted.md` for configuring it, connecting Claude Code /
Claude Desktop, and rotating the key. What follows below is the
standalone CLI (`python -m portolan.mcp_server --http`), useful for a
smoke test or a deployment shape that isn't the bundled web app.

```bash
export PORTOLAN_MCP_ORG_KEY=some-long-random-string   # required beyond localhost
python -m portolan.mcp_server --http --host 0.0.0.0 --port 8000
```

Clients authenticate with `Authorization: Bearer <PORTOLAN_MCP_ORG_KEY>`
against `http://<host>:<port>/mcp`. A request without that header (or
with the wrong key, or with more than one `Authorization` header) gets
`401` before it reaches the MCP protocol layer.

**Fix wave 2026-09-12 (review finding F2).** `PORTOLAN_MCP_ORG_KEY` is
now required for `--http`: `build_http_app` raises instead of building
an app when it's unset, and this CLI reports that as a one-line error
rather than a traceback. The previous behavior -- `--http` still running
with a warning and no real auth -- was proven exploitable end to end (a
running instance with no key served `open_note`'s full `body_md` and
claim quotes, i.e. PHI, to anyone) and is gone. There is no way to run
`--http` without a real key anymore; use `--stdio` (the default) for a
local smoke test instead, which needs no key because the process
boundary is its authentication.

`Settings.mcp_org_key` (`portolan/config.py`) now holds this value;
`server.get_org_key()` reads it via `get_settings()`.

## Tests

```bash
createdb portolan_test_mcp
DATABASE_URL=postgresql://localhost/portolan_test_mcp uv run pytest tests/test_mcp_server.py tests/test_mcp_migration.py
```
