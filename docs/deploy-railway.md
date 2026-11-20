# Deploying Portolan on Railway

Short operational runbook (bug hunt 2026-09-18, P1 finding 8: "the
production environment has no infrastructure-as-code and no runbook in
the repo" -- start command, healthcheck, and volume mount lived only in
the Railway dashboard, readable and changeable by one person from
memory). `railway.json` (repo root) now carries what Railway lets a
config file carry; this file covers what doesn't fit there.

**Watch patterns.** `build.watchPatterns` in `railway.json` limits redeploys to pushes that change what the running app can read: `portolan/`, `corpus/`, `04-brand/`, `evals/`, `docker/`, the lockfile, and the config files. Every deploy stops the old container before the new one mounts the volume, which is about 3 minutes of downtime (2026-09-29). Docs, specs, tests and scripts therefore don't redeploy. A new top-level directory the app reads at runtime must be added to that list.

## Services (as of 2026-09-18)

Project `portolan`, two services:

- **Postgres** -- managed Railway Postgres (pgvector). `DATABASE_URL` on
  the `web` service points at it.
- **web** -- this repo's `docker/Dockerfile`, `PORTOLAN_ROLE=web`. Built
  by Railway itself from the Dockerfile (config-as-code:
  `railway.json`'s `build.dockerfilePath`); if the dashboard still has
  its own `RAILWAY_DOCKERFILE_PATH` service variable set from before
  `railway.json` existed, check it agrees (`docker/Dockerfile`) --
  Railway does not need both, but a conflicting value would win over the
  committed config.

No `worker` service is deployed. `docker/entrypoint.sh`'s `worker` role
exists and works (`PORTOLAN_ROLE=worker`) but nothing runs it today --
see the bug hunt's finding 15 before standing one up.

## What `railway.json` sets

- `build.dockerfilePath` -- builds `docker/Dockerfile`.
- `deploy.healthcheckPath` -- `/healthz` (`portolan/web/routes/health.py`):
  runs `SELECT 1` through the app's own connection pool and returns 503
  on failure, so Railway can tell a live app from one where the database
  connection died (bug hunt finding 1) instead of the "Online, 100%
  500s" state that finding reproduced.
- `deploy.restartPolicyType` / `restartPolicyMaxRetries` -- restart on
  failure, up to 5 attempts, rather than Railway's platform default.

## What still lives only in the Railway dashboard

Railway does not expose these as repo config today; check them by eye
each time this doc is read, not just once:

- **The volume.** `web-volume` mounted at `/data` (`RAILWAY_VOLUME_MOUNT_PATH`).
  `STORE_ROOT` must point at that same mount (or a path under it) -- the
  app now refuses to start otherwise when `RAILWAY_VOLUME_MOUNT_PATH` is
  set (`portolan.store.assert_store_root_ready`, bug hunt finding 5).
- **Replica count.** Stays at 1 unless a real reason to scale comes up.
  If it ever goes above 1, finding 3's advisory-lock fix
  (`portolan.db.migrate_with_backup`) is what keeps two replicas
  starting at once from double-applying a migration -- confirm both are
  running the same image tag first.
- **Wait for CI.** GitHub CI no longer runs on push (2026-09-29: the
  workflow is `workflow_dispatch` only), so this setting waits on nothing and
  Railway deploys a push at once. The gate is local: `scripts/ci_local.py`
  plus `.claude/hooks/push_gate.py`. The three jobs in
  `.github/workflows/ci.yml` (`secrets`, `test`, `docker`) can still be run
  by hand; if they are, this setting should wait for all three. Check this setting directly in
  the dashboard's deploy trigger config. `var/ops/deploy_watch_only.sh
  <sha>` also checks it on every deploy: it exits 3 if Railway starts
  deploying the commit while its CI run is still going.
- **Postgres automated backups.** Whether Railway's own managed-Postgres
  backups are enabled, and their retention. Independent of, not a
  replacement for, the pre-migration `pg_dump` this repo now takes to
  `<STORE_ROOT>/backups/` on the volume (see below) -- that one is
  scoped to "right before a migration runs," not a general recovery
  point.

## Environment variables a production deployment needs (names only --
set real values only in Railway's own variable store, never in a file)

Required on `web` regardless of provider:

```
DATABASE_URL
PORTOLAN_ORG_KEY
PORTOLAN_SESSION_SECRET
STORE_ROOT              # must sit under RAILWAY_VOLUME_MOUNT_PATH
LLM_PROVIDER
EXTRACTION_MODEL
COMPARE_MODEL
STRONG_MODEL
EMBED_PROVIDER
```

Provider-specific (see `.env.example` for the full annotated list):

- `LLM_PROVIDER=bedrock` -- `AWS_REGION`, `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY` (`AWS_SESSION_TOKEN` only if the IAM user needs
  it). `AWS_PROFILE` is a local-dev-only convenience and does nothing on
  Railway.
- `LLM_PROVIDER=anthropic` -- `ANTHROPIC_API_KEY`.
- `LLM_PROVIDER=openrouter` -- `OPENROUTER_API_KEY`.

The app logs one line at startup naming the configured provider and
models (no secrets) -- `portolan starting: llm_provider=... extraction_model=... chat_model=...`
in the deploy log is the fast way to confirm which of the above actually
applies without reading `.env` or the dashboard.

## Backup and restore (bug hunt finding 2)

Startup (`PORTOLAN_ROLE=web` and `PORTOLAN_ROLE=migrate`) now takes a
`pg_dump --format=custom` backup to `<STORE_ROOT>/backups/<timestamp>.dump`
on the volume, but only when there is at least a migration pending, and
aborts the deploy if the dump fails -- see `portolan.db.migrate_with_backup`.
Retention: the newest 10 dumps. `pg_dump`/`pg_restore` are picked to match the running server's major version automatically (checked in order: `PG_BIN_DIR`, Homebrew/PGDG install locations, then PATH) -- set `PG_BIN_DIR` only if none of those has a match.

To check what a deploy *would* apply, or to restore, run these from a
Railway shell attached to the `web` service (so `DATABASE_URL` and the
volume are both in scope) -- never against a local DSN pointed at
production:

```sh
python -m portolan.db migrate --dry-run     # lists pending migrations, applies nothing
python -m portolan.db.rehearse              # backup restored beside the live db, compared, dropped
python -m portolan.db restore <STORE_ROOT>/backups/<timestamp>.dump
```

`restore` is destructive (`pg_restore --clean --if-exists`) -- it
replaces the target database's contents with the dump's. Confirm you're
pointed at the database you mean before running it. It is all or nothing:
it checks the dump against the `<dump>.sha256` written at backup time,
reads the whole file before touching the database, and restores in one
transaction, so a truncated or corrupted dump fails and changes nothing.
A dump from before checksums existed has no `.sha256`; restore it with
`--no-verify-checksum`, which is logged as a warning.

`python -m portolan.db.rehearse` never writes to the live database: it
restores into a second database on the same server and compares. The step-by-step
restore runbook (rehearsal, side-by-side restore and cutover, in place) is
`01-spec/restore-drill-2026-09-22.md`.
