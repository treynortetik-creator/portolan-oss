# Build contracts: module boundaries for parallel agents

2026-09-12. Every agent building a module reads this first. Paths, types and function signatures here are the contract. Change the contract here before changing code, and tell the orchestrator.

## Ground rules for every agent
- Python 3.13, `uv`, pydantic v2, psycopg 3 (`portolan.db.connect()`), pgvector. Type hints, docstrings, no prints outside CLIs, absolute dates in comments.
- **Own only your listed paths.** Never edit another module's files. If you need a change in a shared module, write it in your report.
- **No git commands.** The orchestrator commits between waves.
- **Tests must not hit the network.** Use fakes. Every module ships tests under `tests/test_<module>*.py`.
- **Own test database.** Before running DB tests: `createdb portolan_test_<yourmodule>` and run pytest with `DATABASE_URL=postgresql://localhost/portolan_test_<yourmodule>`. Never use `portolan_test` (shared) while other agents run.
- **Every model call goes through `portolan.llm.LLMClient`** (contract below) and therefore through metering. No direct SDK calls elsewhere.
- Lint: `uv run ruff check <your paths>` clean; `uv run ruff format <your paths>`.
- Fictional names only in examples and tests (see `corpus/truth-table.yaml`).
- **Transaction rule (fix wave 2026-09-12, review finding F15).** Each stage function commits its own work — `intake`, `apply_filter_decision`, `ledger.record`, `write_changelog_row`, every merge/notes writer — and the pipeline treats `sources.extraction_status` as the state machine, never wrapping stages in one transaction. This was inconsistent across the wave A modules (some writers committed internally, `lint/report.py`'s `write_changelog_row` didn't and left it to the CLI) before this fix wave made it uniform; it is the rule going forward, documented here rather than refactored into something else because the wave B pipeline (`portolan/pipeline.py`) already composes stages this way — a stage that fails after an earlier stage committed is a resumable, partially-applied source (retried from wherever `extraction_status` says it got to), not a rolled-back no-op. Don't add a wrapping transaction across stages; if a stage's own internal writes need to be atomic, that's a transaction scoped to that one function, not the caller's problem.

## Shared config: `portolan/config.py` (owned by wave A1)
```python
class Settings(BaseSettings):   # pydantic-settings, reads .env and environment
    database_url: str
    tenant: str = "dev"
    store_root: Path = Path("var/store")        # local file store standing in for S3
    llm_provider: Literal["agent_sdk", "bedrock", "fake"] = "agent_sdk"
    extraction_model: str = "claude-haiku-4-5"  # or a Bedrock id
    compare_model: str = "claude-haiku-4-5"
    strong_model: str = "claude-sonnet-5"
    embed_provider: Literal["fastembed", "titan", "fake"] = "fastembed"
    aws_region: str = "us-east-1"
settings = Settings()
```

## LLM access: `portolan/llm/` (wave A1)
```python
class LLMResult(BaseModel):
    text: str | None
    data: dict | None            # when a json_schema was requested
    input_tokens: int; output_tokens: int; cached_tokens: int; model: str

class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, json_schema: dict | None = None,
                 model: str | None = None, surface: str, max_tokens: int = 4096) -> LLMResult
```
Implementations: `AgentSDKClient` (local, Claude via the Claude Agent SDK; JSON enforced by validating the response against `json_schema`, retry once on failure), `BedrockClient` (wraps `portolan.extract.backends` machinery), `FakeLLMClient(responses)`. Factory: `get_llm() -> LLMClient` from settings. All implementations record a `UsageEvent` via `portolan.metering` with `surface` as given.

**Extraction robustness (fix wave 2026-09-12).** Every real backend (`LLMClientBackend`, `BedrockConverseBackend`, `AnthropicBedrockBackend`) now does one strict attempt, one retry with the validation errors appended on failure, then `ExtractionResult.parse_lenient(data)` as the last resort: it coerces `YYYY-MM`/`YYYY` date strings to a full date and drops (never raises for) any single fact or action that still fails validation, returning `(result, rejected: list[RejectedFact])` — `result.rejected` carries the same list. This is what stopped the first real bench run from crashing outright on a `commitment` fact with no quote and a month-precision `valid_from`. `Fact.type` is `str`, not the closed `FactType` enum (review finding F13): validated against the *effective* type list (base `FactType` names, or base + a customer extension's `extra_types` when the caller passes `context={"effective_schema": merge_customer_extension(extension)}` into `model_validate`/`parse_lenient`), with `quote_required` honoured per-type and `severity_overrides` applied as a post-validation step. Nothing wires a real `CustomerExtension` into extraction yet — every current caller validates against the base list only — so this is the schema-layer half of F13; wiring a specific customer's extension through is future work.

`portolan/extract/backends.py` gains an `LLMClientBackend(client, model: str | None = None)` so extraction runs through the same client, with `model` passed straight through to every `complete()` call (review finding F17, fix wave 2026-09-12 — the constructor originally took only `client`, so it couldn't select a model at all, and `bench/__main__.py` had a bare `except TypeError` retry working around the missing parameter; both are fixed now).

**`LLMClientBackend(client, model=None, surface="extract")`** (polish wave 2026-09-12): `surface` is now a third constructor argument rather than a hardcoded string, so a caller that needs a different metering surface (`portolan.audit.second_pass`'s production-audit pass, surface `"audit"`) no longer needs its own near-duplicate subclass. `AuditBackend` is kept as a one-line alias only because `portolan.audit.__main__`/`__init__` still import it by name.

## Store: `portolan/store.py` (wave A2)
```python
class Store(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None
    def get(self, key: str) -> bytes
    def exists(self, key: str) -> bool
    def url(self, key: str) -> str      # local path or signed URL
class LocalStore(Store)   # under settings.store_root
class S3Store(Store)      # boto3, lazy client
```

## Source model and intake: `portolan/ingest/` (wave A2)
```python
class RawSource(BaseModel):          # what a connector hands to intake
    source_id: str; source_type: SourceType; source_date: date
    participants: list[Participant]  # name, email|None, role|None
    account_hints: list[str]         # crm ids or names the connector already knows
    text: str                        # raw text; connector-specific formats normalised to text
    original_bytes: bytes | None; original_content_type: str | None
    metadata: dict

def intake(raw: RawSource, *, conn, store) -> SourceRow      # writes sources row (status pending), stores original, computes content_hash; idempotent on content_hash via a real unique index (sources_content_hash_uq, 0006, review finding F6) plus INSERT ... ON CONFLICT DO NOTHING RETURNING. A source whose content changed under the same source_id gets a new slug <source_id>--v<n>, with metadata.supersedes_source recording what it versions past (see provenance-chain.md "Source versioning").
def prepare_text(raw: RawSource) -> PreparedText              # strips quoted email text and boilerplate, injects locators per source type (settled formats in provenance-chain.md, verified against corpus/out/manifest.json — fix wave 2026-09-12, review finding F8); returns text + locator index. PreparedText carries stripping_skipped: bool (F1) — set when quote-stripping would have returned empty text for non-empty input and prepare_text fell back to the unstripped original instead.
```
Filter (same wave):
```python
class FilterDecision(BaseModel): include: bool; reason: str; matched_entities: list[str]; sampled: bool
def customer_filter(raw: RawSource, prepared: PreparedText, registry: Registry, *, sample_rate: float) -> FilterDecision
```
Rules in order: junk (auto-replies, calendar invites, newsletters) → participant email domain or address in registry → account_hints → registry scan of text (names, aliases, products, signal words) → excluded, with `sampled=True` on a seeded 1 to 2 percent of exclusions — **junk exclusions included** (fix wave 2026-09-12, review finding F9: junk used to skip sampling entirely, so a junk misclassification, like the out-of-office body-match false-positive that same finding fixes, had no audit path at all). **Carve-out (A2, 2026-09-12):** the signal-word rule is skipped when every participant resolves to an `internal_person`, so an internal ops thread about "deliveries" with no customer or product named is excluded and left for the sample audit (truth table F07). Every decision updates `coverage_ledger` (connector = source_type, day = source_date) — as of the fix wave, that write lives in `portolan.pipeline`'s filter stage (via `portolan.ingest.ledger.record`), not in `customer_filter` or `apply_filter_decision` itself (review finding F16: `apply_filter_decision` only ever updates the `sources` row, so there is exactly one place in the codebase that records a filter decision to the ledger).

## Registry: `portolan/registry/` (wave A3)
```python
# Entity.id == registry_entities.crm_id for CRM rows (e.g. "A-1003"); the meta slug for products/competitors/internal people. Truth-table account slugs are aliases only. Anything scoring against corpus/truth-table.yaml maps slug ↔ crm_id through its `accounts` list.
class Entity(BaseModel): id: str; kind: EntityKind; canonical_name: str; crm_id: str | None; parent_id: str | None; account_kind: str | None; email: str | None; email_domain: str | None; aliases: list[str]
class Registry:
    @classmethod
    def load(cls, conn) -> "Registry"
    def import_crm(self, conn, accounts_csv: Path, contacts_csv: Path) -> ImportReport   # dedup rules per spec §3; writes registry_entities + registry_aliases; review_queue rows for uncertain merges
    def resolve(self, mention: str, *, kind_hint: EntityKind | None = None, context_ids: list[str] = ()) -> Resolution
    def ancestors(self, entity_id: str) -> list[str]     # parent chain, nearest first
    def by_email(self, email: str) -> Entity | None
    def by_domain(self, domain: str) -> list[Entity]
    def scan(self, text: str) -> list[Match]              # fast multi-pattern scan over names and aliases (Aho-Corasick or compiled regex), case-insensitive, word-boundary
    # Match must carry at least: entity_id, canonical_name, kind, start, end. Entity must carry at least: id, kind (A2 codes against these).
class Resolution(BaseModel): status: Literal["resolved", "ambiguous", "unknown"]; entity_id: str | None; candidates: list[str]; attach_to: str | None   # ambiguous → shared parent if any
def add_alias(conn, entity_id, alias, source: str) -> None
def queue_unknown(conn, mention: str, source_slug: str, context: dict) -> None
```

## Embeddings, index, search: `portolan/search/` (wave A4)
```python
class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]
class FastEmbedEmbedder   # BAAI/bge-large-en-v1.5, 1024-dim, lazy load
class TitanEmbedder       # Bedrock, lazy client
class FakeEmbedder        # deterministic hash-based vectors, 1024-dim

def chunk_note(body_md: str, max_chars: int = 1200) -> list[Chunk]       # heading-aware, records nearest heading
def index_note(conn, embedder, note_slug: str) -> int                     # re-chunk and re-embed only if notes.body_md hash changed; returns chunks written
def reindex_all(conn, embedder) -> IndexReport

class SearchHit(BaseModel): note_slug: str; title: str; score: float; via: list[Literal["fts", "vector"]]; best_chunk: str; heading: str | None; updated: date; source_slugs: list[str]
def search(conn, embedder, query: str, *, k: int = 12, expand: int = 0, types: list[str] | None = None, since: date | None = None, note_slugs: list[str] | None = None, entity_slugs: list[str] | None = None) -> SearchResult
```
Fusion: RRF with k=60, vector weight 1.2, fts weight 1.0; title match boost; recency multiplier from `notes.updated` with absolute exponential decay (half-life 180 days, floor 0.85); `expand` follows `links` from the top hits; `SearchResult.control_ok` runs a control query when zero hits. Full text uses `notes.search` and `chunks` tsvector with `ts_rank_cd`.

**`note_slugs`/`entity_slugs`** (polish wave 2026-09-12, review G16): restrict the search *space* itself (the underlying SQL, not a post-hoc filter on the top-k global ranking) to a given set of notes, or to every note structurally linked (`body`/`parent`) to a given set of entity notes. Added for the merge candidate gate's cross-note half — see `portolan/merge/gate.py`'s `gate()` below.

## Notes and claims: `portolan/notes/` (wave B1, but the writer helpers are needed by A4 tests: A4 uses raw SQL inserts in its tests, not this module)
```python
class Claim(BaseModel): id: int | None; note_slug: str; fact_type: str; claim_text: str; source_slug: str; locator: str; quote: str | None; speaker: str | None; valid_from: date | None; valid_to: date | None; recorded: datetime; superseded_at: datetime | None; severity; sentiment; confidence
def render_note(note: NoteRow, claims: list[Claim]) -> str     # markdown with frontmatter, synthesis section, and a claims section where every claim line ends with ^[source-slug @ locator]
def write_note(conn, note: NoteRow, claims: list[Claim], *, changed_by: str, reason: str) -> None  # versions the previous body into note_versions; updates links from [[wikilinks]] and provenance
def slugify(title: str) -> str
```

## Merge: `portolan/merge/` (wave B1)
```python
def merge_fact(conn, llm, embedder, registry, fact: Fact, source: SourceRow) -> MergeOutcome   # candidate gate via search(); compare only when a candidate scores above threshold; outcomes new|same|extends|contradicts; writes claims with two clocks; supersedes on contradiction when dated later
# Polish wave 2026-09-12 (review G16): `portolan.merge.gate.gate()` grew optional `conn`/`entity_slugs`/`exclude_note_slugs` keyword args so the candidate gate can search other notes linked to the same account, not just the fact's own home note (backward compatible: omitting them is the pre-G16, home-note-only gate). `merge_fact` above (portolan/merge/facts.py, not owned by this fix) needs a small change to actually pass them -- `conn=conn, entity_slugs=[the fact's account entity/note slug], exclude_note_slugs=[home_note.slug]` -- to exercise this in the live pipeline; see `tests/test_merge_gate.py` for proof the mechanism itself works.
def merge_extraction(conn, llm, embedder, registry, result: ExtractionResult, source: SourceRow) -> MergeReport
```
Entity notes (`account`, `contact`, `product`, `competitor`) are created from the registry on first reference; `signal` notes are per theme-able fact cluster (v1: one signal note per fact type per account, appended to); `action` notes from `result.actions`.

## Pipeline runner: `portolan/pipeline.py` (wave B2)
`run(conn, *, connector: Connector, llm, embedder, registry, store, settings, sink: UsageSink | None = None) -> RunReport`. Stages as functions with retries (3, backoff) and a `dead_letter` table write on final failure (add the table in migration `0004_dead_letter.sql`, owned by B2; see the migration ledger). Partial handling per segment. CLI: `python -m portolan.pipeline --connector file --path corpus/out`.

**Extraction fan-out** (polish wave 2026-09-12, spec sec 5.1 "Extraction fans out; merge and write are single-writer"): `run()` now processes sources in fixed-size windows (`PORTOLAN_EXTRACT_CONCURRENCY` env var, default 4 -- flagged here for `Settings` to absorb properly, since this fix does not own `portolan/config.py`), fanning the segment-extraction calls for one window out across a `ThreadPoolExecutor` while fetch/intake/prepare/filter and merge/write/index stay strictly sequential, in source order, on the main thread. `sink` (task 7): defaults to `portolan.metering.PostgresSink(conn)` when not given, so a real run's metering lands in `usage_events`; every test passes its own, in-memory.

## Connectors: `portolan/connectors/` (wave B2 for `FileConnector`; wave D2 for the rest)
```python
class Connector(Protocol):
    name: str
    def list_sources(self, since: date | None) -> Iterable[SourceRef]
    def fetch(self, ref: SourceRef) -> RawSource
class FileConnector(Connector)   # reads corpus/out layout: calls/*.txt, tickets/*.json, email/*.eml, crm/*.csv, rss/*.json
```

## Lint: `portolan/lint/` (wave B3) — `run_lint(conn, store) -> LintReport` and a markdown renderer; checks per spec §6.1.

## Bench: `portolan/bench/` (wave B4) — `score_extraction(manifest, truth_table, results) -> ExtractionScore`; `score_graph(conn, truth_table) -> GraphScore` (expectations block of the truth table); CLI `python -m portolan.bench --backend agent_sdk|fake --corpus corpus/out`.

## Report, audit, consolidation: `portolan/report/`, `portolan/audit/` (wave C1, C2) — interfaces defined in `surfacing-and-weekly-report.md` and `quality-loop.md`; the C agents propose signatures in their reports.

## Web and MCP: `portolan/web/`, `portolan/mcp_server.py` (wave C3, C4) — FastAPI app factory `create_app(settings) -> FastAPI`; routes: `/chat`, `/notes/{slug}`, `/sources/{slug}`, `/originals/{slug}`, `/graph`, `/reports`, `/coverage`, `/edit`; MCP tools: `search`, `expand`, `open_note`, `open_source`, `add_standing_question`, `coverage`.

## Migration ledger (append here before adding a file)
| File | Owner | Purpose |
|---|---|---|
| 0001_init.sql | Phase 0 | the schema |
| 0002_index_state.sql | A4 | note body hash for incremental indexing |
| 0003_standing_questions.sql | C4 | standing questions per reader for the MCP and the weekly report |
| 0004_dead_letter.sql | B2 | pipeline dead-letter queue |
| 0005_themes_reports.sql | C1 | theme membership, report rows, standing-question answers |
| 0006_review_fixes.sql | fix wave 2026-09-12 | unique index on `sources.content_hash` (F6); see also 0001's FK-add fix in place (F7) |
| 0007_claim_citations.sql | fix wave 2 2026-09-12 | `claim_citations(claim_id, source_slug, locator, quote)` -- a `same` merge verdict records a corroborating source here instead of a duplicate `claims` row (review G4) |
| 0008_theme_members_clocks.sql | fix wave 2 2026-09-12 | `theme_members.attached_on` -- the wall-clock attach day, a second clock alongside `added` (world date), so `theme_trend` buckets by the one it means to (review G8) |
| 0009_sources_metadata.sql | polish wave 2026-09-12 | No new column -- documents `sources.metadata`'s expanded shape: `raw` now holds the connector's own `RawSource.metadata` verbatim (alongside the existing `supersedes_source`), so `portolan.audit.second_pass` can reconstruct a `RawSource` losslessly instead of re-deriving one from stored bytes |
| 0010_conversations.sql | UI wave 1, 2026-09-12 | `conversations`/`messages` for the two-pane chat surface -- hard-deletable, unlike the rest of this schema (a reader's own scratch space, not provenance data) |
| 0011_theme_resolutions.sql | theme identity/resolution wave 2026-09-13 | `theme_resolutions(theme_slug, resolution_claim_id, resolved_reason, resolved_week)` -- one row per theme resolved by an explicit claim (task item 2(a)); the quiet-weeks resolution path is recomputed live and stores nothing |
| 0012_theme_resolution_reopening.sql | theme identity wave code review fix 2026-09-13 | `theme_resolutions.reopened_week` (nullable) -- review blocker 1 / "Resolution rule" item 3: a claim-based resolution is reopen-able (`>= THEME_REOPEN_MIN_MEMBERS` new members after `resolved_week`) instead of silently permanent; the row itself is kept as history, never a second row per theme |
| 0013_theme_resolution_history.sql | theme review fix wave 2026-09-13 | `theme_resolutions` gets a synthetic `id` PRIMARY KEY (was `theme_slug`) plus a partial unique index keeping at most one ACTIVE (`reopened_week IS NULL`) row per theme -- should-fix 5/6: a theme can now reopen and be claim-resolved again any number of times (history rows, not a single overwritten flag), and a merge that must reconcile two active rows closes the later one out instead of deleting it |
| 0014_theme_resolution_confirmation.sql | second theme fix wave 2026-09-13 | `theme_resolutions.confirmed_by`/`confirmed_at` (both nullable) -- nit 19 (`review-theme-fix-wave-2.md`): a human confirming a claim-based resolution via `/review` (`portolan.report.themes.confirm_theme_resolution`) now stamps who and when on the resolution row itself, instead of only flipping the `review_queue` row's own status; confirmation is advisory (it does not gate whether the resolution already took effect), and both columns stay `NULL` until a human actually clicks confirm |
| 0015_theme_resolution_status.sql | fifth theme fix wave 2026-09-14 | Decisions log "withdraw content-based automatic resolution": `theme_resolutions.status` (`proposed`/`confirmed`/`rejected`/`superseded`, existing rows backfilled `confirmed`), `.evidence` (nullable, the judge's quoted sentence), `resolution_claim_id` made nullable (the mechanical silence path names no claim), and a unique index on `(theme_slug, resolution_claim_id)` so a `(theme, claim)` pair is judged at most once ever |
| 0016_theme_resolution_claim_index_open_only.sql | review-theme-wave-5-judge.md fix wave 2026-09-14 | Blocker 2: `uq_theme_resolutions_theme_claim` (migration 0015) rescoped to `reopened_week IS NULL` (open rows only) -- the old, all-rows-ever version collided with this design's own blessed "one claim can propose several themes" shape the first time two such themes merged (`_carry_theme_resolution` closes the later-resolved side out, then moves every row onto the surviving slug, landing a closed loser and an open winner on the same `(theme_slug, resolution_claim_id)` key), raising `UniqueViolation` out of a module with no exception handling and taking the weekly report down with it. The (theme, claim)-judged-once-ever invariant was, and remains, enforced at the application layer (`_check_theme_resolutions`'s `already_judged` full-history check) independent of this index |
| 0017_theme_judge_cache_and_fairness.sql | review-theme-wave-5b.md fix wave 2026-09-14 | Two tables, neither ever read as evidence of anything by itself -- pure cost/fairness optimizations over `_check_theme_resolutions`'s own judge loop. `theme_judge_verdicts(theme_slug, candidate_claim_id, ...)`: remembers a `false` resolution-judge verdict was reached for one (theme, candidate) pair, so an unchanged pair is skipped rather than re-judged (and re-paid) on every single run -- a `true` verdict needs no cache, it already has its own durable `theme_resolutions` row. `theme_judge_cursor`: a singleton row naming the theme_slug that led the judge queue last run, so the round-robin cap (`THEME_JUDGE_MAX_CALLS`) rotates which themes get judged first instead of starving the same alphabetically-early ones every week |
| 0018_theme_judge_prompt_hash_and_retries.sql | review-theme-wave-5c.md fix wave 2026-09-14 | Should-fix 3: replaces `theme_judge_verdicts`'s two-column staleness key (`theme_member_count`, `candidate_text_hash` -- a member-count proxy that a theme rename, an in-place member edit, or an equal-count member swap could all defeat) with one `prompt_state_hash`, over exactly what the judge prompt shows (theme name + every member's own text + candidate text) -- any change to any of it forces a re-judge. Should-fix 4: adds `theme_judge_unverified(theme_slug, candidate_claim_id, prompt_state_hash, retry_count, ...)`, counting consecutive `resolves: true`-but-unverifiable-evidence verdicts per pair so one is re-judged (not cached `false`, which would bury a real resolution) up to `THEME_JUDGE_RETRY_LIMIT` times before finally being cached. Neither table is ever read as evidence of anything by itself either |
| 0019_reports_judge_health.sql | review-theme-wave-5c.md fix wave 2026-09-14 | Should-fix 7: `reports.judge_health` (nullable jsonb) -- a snapshot of `judge_calls`/`judge_failures`/`judge_cap_hit` plus a derived `outage` flag (every call this run failed, not merely some), written once per report alongside the `reports` row itself (`build_weekly`) so an operator can see judge health after the fact across weeks, not just by re-reading that week's own report body. Never read as evidence of anything about the graph -- purely an operational signal |
| 0020_context_note_type.sql | company-context wave 2026-09-17 | Owner decision 2026-09-17 ("Company context page"): adds `context` to the `notes.type` closed list (drop/re-add of the auto-named `notes_type_check` CHECK, verified against the running schema -- Postgres has no `ALTER CONSTRAINT ... ADD VALUE`). One `context`-typed note per tenant, slug `company-context`, seeded at app startup (`portolan.context.seed_company_context`, called from `portolan.web.app`'s lifespan, after migrations) and editable through the existing `/notes/{slug}/edit` flow -- no new route, no new table |
| 0021_claims_context_stamp.sql | company-context wave 2026-09-17 | `claims.company_context_updated_at` (nullable timestamptz, no default/backfill) -- the company-context note's own `notes.updated` at the moment the pipeline run that produced this claim loaded it (`portolan.context.load_company_context`), so a later reader can tell which version of the context a fact was extracted under. Threaded as an optional kwarg through `portolan.merge.pipeline.merge_extraction` -> `portolan.merge.facts.merge_fact` -> `_build_claim`, default `None` everywhere so every pre-existing caller/test is unaffected |

## Company and account context: `portolan/context.py` (company-context wave, 2026-09-17)
```python
COMPANY_CONTEXT_SLUG = "company-context"
COMPANY_CONTEXT_BODY_CHAR_CAP = 2500
ACCOUNT_CONTEXT_PARAGRAPH_CAP = 400

class CompanyContext(BaseModel): text: str; updated: datetime

def load_company_context(conn) -> CompanyContext | None            # None + one warning logged when the note is missing or empty
def render_company_context_block(text: str) -> str                # "## Company context\n<text>"
def build_account_context(conn, registry_context: list[RegistryContextEntry]) -> dict[str, str]   # {registry_id: registry line + capped first paragraph of that account note's synthesis}, ACCOUNT entries only
def render_account_context_block(entries: dict[str, str]) -> str   # "" when entries is empty
def seed_company_context(conn) -> bool                             # idempotent; never overwrites an existing note; returns whether it created one
```
Deliberately kept out of `portolan.extract.prompt` (that module's own docstring: "pure rendering... unit-tested without touching a model") -- this module does the one piece of database I/O every caller (`portolan.pipeline`, `portolan.report.themes`, `portolan.web.answer`, `portolan.bench`) needs before it can hand `prompt.py` a plain string. `prompt.build_system_prompt` grew a `company_context: str | None = None` kwarg (rendered before the vocabulary block); `SourceEnvelope` grew an `account_context: dict[str, str]` field, rendered by `prompt.build_user_message` as its own "## Account context" section after "## Registry context". `themes.RESOLUTION_JUDGE_SYSTEM` and `answer.ANSWER_SYSTEM_PROMPT` stay constants; both call sites append `render_company_context_block(...)` to them at call time when a context note exists -- so every existing test asserting the bare constant on a context-free test database is unaffected.

## Infra: `infra/` (wave D1) — CDK Python app; not deployed until the AWS account exists.
