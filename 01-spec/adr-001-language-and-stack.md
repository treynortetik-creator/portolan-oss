# ADR-001: language and stack

2026-09-12. Status: accepted by the orchestrator, per the maintainer's instruction to think about the language per component rather than default.

## The question
Which language and framework for each part of Portolan: the pipeline, the search layer, the web app (chat and viewer), the MCP server, the infrastructure code, and the connectors.

## What matters here
- **One person maintains it, mostly through agents.** Every extra language, build tool or process is one more thing that can break a deploy, which is the stated top fear. Fewer moving parts beats a marginally better UI.
- **The heavy logic is text and data**: extraction, entity resolution, merging, embeddings, ranking, lint. That is Python's home ground, and the reference implementation is already Python.
- **The UI is a chat, a note viewer, a graph view and a document viewer.** It is read-heavy, not an app with complex client state. Server-rendered pages with a few JavaScript islands cover it.
- **Everything runs in AWS**; infrastructure as code needs to be reviewable by agents and the owner.

## Decision

| Component | Choice | Why | Rejected |
|---|---|---|---|
| Pipeline, registry, merge, lint, bench, report | **Python 3.13** | pydantic, boto3, psycopg, fastembed, the MCP SDK and the Anthropic SDK are all first-class; the existing code is Python; agents are strongest here | Go or Rust for throughput: the bottleneck is model calls, not CPU |
| Search and index | **Postgres 17 with pgvector and tsvector**, Python on top | One database per customer holds vectors, full text, notes, registry, ledger, usage. No separate search service | OpenSearch, LanceDB: a second system to run for no gain at this scale |
| Web app: chat, viewer, graph, edits | **FastAPI + Jinja2 templates + HTMX, with three JavaScript islands** (force graph via a CDN library, transcript seek-to-timestamp, PDF.js) served from the same process as the API | Single language, single container, single test suite; streaming chat works over server-sent events; the API is there if a React front end is ever wanted | Next.js or React: better chat polish, but a second build pipeline, CORS, and two languages for one maintainer |
| MCP server | **Python, official MCP SDK** | Same process family as the search code it wraps | TypeScript MCP SDK: would split the codebase |
| Infrastructure | **AWS CDK in Python** | Keeps one language; typed constructs; agents read and review it as code; synthesises to CloudFormation which AWS runs | Terraform: fine, but HCL is a second language; CloudFormation YAML: unreadable at size |
| Connectors | Python clients, one small module each, with a `FileConnector` that reads exports | Same reasons | |
| Local model access | Claude Agent SDK during local development (uses the Claude Code login, no API key); Bedrock in AWS | Lets the bench and the corpus generator run today | |
| Local embeddings | fastembed `bge-large-en-v1.5` (1024-dim) so the schema's `vector(1024)` matches Titan V2 in AWS | No schema difference between dev and prod | |

## Consequences
- One `uv`-managed Python project, one Dockerfile, one CI. The web app and the pipeline are two entry points in the same image.
- If the chat UI later needs more than HTMX gives, add a React front end against the existing API; nothing has to be rewritten.
- The JavaScript islands are the only non-Python code and are pinned CDN libraries, not a build step.
