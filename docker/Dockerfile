# Portolan container image. The same image runs four roles
# (PORTOLAN_ROLE=web|worker|migrate|test, see docker/entrypoint.sh) -- so
# the containerised path and the CI path are provably the same path (spec
# sec 11: "Comprehensive CI runs the full test suite in Docker on every
# release").
#
# Fix wave 2026-09-12 (review finding F4): this image used to have no
# entrypoint that served anything at all -- `CMD ["uv","run","pytest",
# "-q"]` ran the test suite and exited, while the ECS task's ALB
# health-checked port 8000 against nothing. See docker/entrypoint.sh.
#
# Bug hunt 2026-09-18 (ops review):
# - P1 finding 9: the base image and the uv installer image were both
#   floating tags (`python:3.13-slim`, `ghcr.io/astral-sh/uv:latest`), so
#   a release between CI's build and Railway's own build of the same
#   commit could ship two different images from one green check. Both
#   pinned by digest below (resolved 2026-09-18 against the exact tags
#   this project already used -- `python:3.13-slim`,
#   `ghcr.io/astral-sh/uv:0.11.3`, the version `.github/workflows/ci.yml`
#   already pins the `uv` CLI to on the runner). To refresh either
#   deliberately: `docker buildx imagetools inspect <image>:<tag>` and
#   swap in the new digest -- never drop back to a floating tag.
# - P0 finding 2: `postgresql-client-17` (matching the pgvector/pgvector
#   `pg17` this project runs everywhere -- docker-compose.yml, CI) is
#   installed below so `pg_dump`/`pg_restore` (`portolan.db.
#   backup_database`/`restore_database`, `python -m portolan.db`) exist
#   in the image that startup migration path actually runs in. Debian's
#   own `postgresql-client` package tracks whatever major version shipped
#   with the base image's Debian release, which can trail the server's --
#   pg_dump refuses to dump a server newer than itself, so the version is
#   pinned explicitly via the official PGDG apt repo instead.
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

# Install uv (the project's package manager) via its official installer image.
COPY --from=ghcr.io/astral-sh/uv:0.11.3@sha256:90bbb3c16635e9627f49eec6539f956d70746c409209041800a0280b93152823 /uv /uvx /usr/local/bin/

WORKDIR /app

# postgresql-client-17 and -18: pg_dump/pg_restore for the pre-migration backup;
# Railway Postgres runs 18 (deploy crash 2026-09-18 18:48 MST), local and CI run 17,
# and portolan.db picks the client matching the server major at startup
# and `python -m portolan.db restore` (bug hunt 2026-09-18, P0 finding
# 2). Installed from the PGDG apt repo, pinned to major version 17,
# rather than Debian's own `postgresql-client` package (see the header
# comment above for why the version has to be pinned explicitly).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && install -d -m 0755 /usr/share/postgresql-common/pgdg \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt ${VERSION_CODENAME}-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client-17 postgresql-client-18 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first so dependency installation is cached
# independently of application code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# .dockerignore (review finding F4) keeps .env, .git, var/, the large
# synthetic corpus, this repo's own memory/agent tooling, and the dev-only
# scripts/ directory (reason in .dockerignore) out of this layer -- COPY . .
# only ever sees what .dockerignore lets through.
COPY . .
RUN uv sync --frozen
RUN chmod +x docker/entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

# tests/test_infra.py and tests/test_infra_deploy.py import aws_cdk,
# which is jsii-backed and spawns a real `node` child process just to
# import it -- FileNotFoundError: 'node' at collection, since this image
# deliberately has no Node (CDK/infra is exercised by the `test` job on
# the runner instead, which does have Node; see .github/workflows/ci.yml).
# Do not install Node here to make these pass in-image.
ENTRYPOINT ["docker/entrypoint.sh"]

# Bug hunt 2026-09-18 (ops review, P1 finding 4): no CMD -- the old
# `CMD ["uv","run","pytest","-q"]` was the image's fallback whenever
# PORTOLAN_ROLE was unset or dropped (a new Railway environment, a
# recreated service, a variable edit), so a container meant to serve
# traffic silently ran the test suite and exited instead. docker/
# entrypoint.sh no longer falls through to CMD for any input -- an
# unset/empty PORTOLAN_ROLE now defaults to `web` there directly, and
# running the suite in-container is the explicit `PORTOLAN_ROLE=test`
# role. A CMD here would never be reached by any of entrypoint.sh's
# branches, so there is deliberately none, rather than leaving a value
# that reads as live but isn't.

# HEALTHCHECK (review finding F4 follow-up / bug hunt 2026-09-18 P2
# finding 12): only meaningful for the `web` role, which is the only one
# that serves 8000 -- a no-op (exit 0) for every other role rather than
# reporting `worker`/`migrate`/`test` containers unhealthy for not
# serving HTTP. Points at the same /healthz route
# (`portolan/web/routes/health.py`) Railway's own healthcheck now uses
# (railway.json).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["sh", "-c", "test \"${PORTOLAN_ROLE:-web}\" != web || curl -fsS http://127.0.0.1:8000/healthz"]
