#!/bin/sh
# Portolan container entrypoint. Branches on PORTOLAN_ROLE (review
# finding F4: "The image has no entrypoint that serves anything --
# PORTOLAN_ROLE=web|worker is read by zero lines in the repo (grepped),
# so the ECS task runs the test suite and exits while the ALB
# health-checks port 8000").
#
# Bug hunt 2026-09-18 (ops review, P1 finding 4): an UNSET or dropped
# PORTOLAN_ROLE used to fall through to `exec "$@"` -- the image's own
# CMD, which ran the test suite and exited while nothing served port
# 8000 (the F4 outage, recurring, guarded only by a dashboard value).
# An unset/empty role defaults to `web` below and is exported, so
# `python -m portolan.web` sees it and binds 0.0.0.0 rather than loopback
# (tests/test_docker.py). There is no path that falls through to CMD at
# all -- every branch either execs something that serves a role, or exits
# non-zero with a clear message (the `*` case). The image's CMD is
# correspondingly never the test suite either
# (docker/Dockerfile) -- running the suite in-container now requires the
# explicit `PORTOLAN_ROLE=test` this script reads below, the same way
# `web`/`worker`/`migrate` are explicit.
set -eu

PORTOLAN_ROLE="${PORTOLAN_ROLE:-web}"
export PORTOLAN_ROLE

case "$PORTOLAN_ROLE" in
  web)
    exec python -m portolan.web
    ;;
  worker)
    # A polling loop over the configured connector's pipeline run, not
    # yet the persistent SQS-consuming worker infra/portolan_infra
    # provisions an autoscaling ECS service for (portolan/pipeline.py is
    # a concurrent fix wave's file, out of this wave's owned paths) --
    # said so here rather than silently pretending this is the finished
    # thing. Good enough to prove PORTOLAN_ROLE=worker does real,
    # repeated ingest work; a real queue consumer is the next wave's job.
    #
    # Bug hunt 2026-09-18 (ops review, P2 finding 15): a failing cycle
    # used to be swallowed by `|| true` -- indistinguishable in the logs
    # from a cycle that succeeded. Dropped; the exit code is logged
    # instead, and the loop keeps running (this role is not deployed
    # anywhere today, per the same finding -- a customer-visible
    # "worker keeps crash-looping" alert is a later wave's job, once it
    # is).
    echo "portolan worker: polling '${PORTOLAN_CONNECTOR:-file}' every ${PORTOLAN_WORKER_POLL_SECONDS:-300}s" >&2
    while true; do
      if python -m portolan.pipeline --connector "${PORTOLAN_CONNECTOR:-file}"; then
        :
      else
        echo "portolan worker: cycle failed (exit $?)" >&2
      fi
      sleep "${PORTOLAN_WORKER_POLL_SECONDS:-300}"
    done
    ;;
  migrate)
    # Bug hunt 2026-09-18 (ops review, P0 finding 2 / P1 finding 3): was
    # a bare `apply_migrations` call with no backup and no advisory lock.
    # `python -m portolan.db migrate` is the same path `portolan.web.
    # app`'s own startup now takes -- an advisory lock (two replicas
    # migrating at once can no longer race) and a pg_dump backup before
    # anything is applied.
    exec python -m portolan.db migrate
    ;;
  test)
    # The explicit opt-in to run the test suite in-container (spec sec
    # 11: "Comprehensive CI runs the full test suite in Docker on every
    # release") -- see the P1 finding 4 note above for why this is no
    # longer the CMD-fallthrough default.
    exec uv run pytest -q --ignore=tests/test_infra.py --ignore=tests/test_infra_deploy.py
    ;;
  *)
    echo "portolan entrypoint: unknown PORTOLAN_ROLE='${PORTOLAN_ROLE}' (expected web, worker, migrate, or test)" >&2
    exit 1
    ;;
esac
