# Portolan (working name)

A customer-knowledge graph engine with provenance. Portolan reads the places customer knowledge already sits (call transcripts, CRM exports, emails, support tickets, documents, public news feeds), extracts atomic facts, and merges them into a graph of small linked notes about accounts, contacts, products, signals and themes. Every claim carries a citation to the exact spot in the original source it came from: the call at its timestamp, the email, the file and page.

An organization queries the graph through one MCP endpoint or a small web app. Scheduled ingest keeps it current; a linter and a quality audit keep it honest; a consolidation pass keeps it readable.

## Design rules

- **Everything traces back.** An answer cites a note, a note cites a source, a source points at the original. A sentence with nothing behind it is not shown. See `01-spec/provenance-chain.md`.
- **Nothing sticky comes from a heuristic.** An automated judgement may *propose* (for example, that a theme is resolved); a human, or a configured period of silence, disposes. See `01-spec/surfacing-and-weekly-report.md`.
- **A human edit is a source.** Hand fixes to the graph are attributed records, not silent overwrites. See `01-spec/quality-loop.md`.
- **Contradictions are kept.** When two notes disagree, both stay with their dates; the newer replaces the older only when the dates settle it, otherwise the pair goes to a review queue.

## Layout

```
portolan/        the engine
  ingest/ extract/ merge/ consolidate/   source -> facts -> notes
  search/                                 fused keyword + semantic search with recency weighting
  notes/ report/ signals/                 note store, weekly report, external signals
  lint/ audit/ registry/                  quality checks and the source registry
  connectors/ crm/                        Salesforce, HubSpot, Gong, Zoom, Clari, RSS, ticket and file connectors
  llm/ mcp_server/ web/ cli/ db/          model routing, MCP door, web app, CLI, Postgres schema
tests/           pytest suite
corpus/          synthetic truth table + generator (all names invented; the bench scores against it)
evals/           chat exam sets
scripts/         bench, data, ops and verification scripts
infra/           AWS CDK stacks (per-customer isolated environments)
docker/          Dockerfile + compose for the containerised test path
docs/            CLI, hosted MCP, Railway deploy
01-spec/         design notes: provenance chain, extraction schema, quality loop, surfacing, stack ADR
04-brand/        wordmark and design tokens used by the web app
FEATURES.md      what a user sees and how to prove each piece works
```

## Stack

Python 3.13, Postgres 17 with pgvector, FastAPI, `uv` for dependencies. Extraction and judging run on cheap models through OpenRouter or a local backend; see `portolan/llm/`.

## Dev setup

```bash
uv sync                                  # installs deps into .venv

brew install postgresql@17 pgvector      # or your platform's equivalent
brew services start postgresql@17
createdb portolan_dev                    # the dev graph (.env.example points at it)
cp .env.example .env                     # fill in your own keys

env -u OPENROUTER_API_KEY -u DATABASE_URL TZ=UTC PGTZ=UTC PORTOLAN_ROLE=test uv run pytest -q
```

The suite creates and drops its own `portolan_test_<pid>` database. Never give the dev graph a name starting with `portolan_test`: the suite treats those as disposable.

To try it end to end on the synthetic corpus, see "Local proof run" in `FEATURES.md`.

## Data rules for contributors

- No real customer data in this repo. The corpus is synthetic by construction.
- `tests/test_no_outside_material.py` fails on data-shaped files outside the synthetic directories. Set `PORTOLAN_FORBIDDEN_NAMES` (comma-separated) to also fail on any names you never want committed.

## Status

Early (v0.1.0). The engine, bench and web app run; the AWS stacks are written but not deployed. "Portolan" is a working name.

## License

MIT. See `LICENSE`.
