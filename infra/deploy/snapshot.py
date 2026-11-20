"""Create an Aurora cluster snapshot and wait for it to become available.

Spec: `01-spec/portolan-spec-v2.md` §11 — "Snapshot every database before
deploy." Run once per customer, immediately before that customer's release
deploys (see `canary.py`, which calls this before each deploy step).

Usage:
    uv run python infra/deploy/snapshot.py \\
        --customer acme --cluster-identifier portolan-acme-database --region us-east-1

    uv run python infra/deploy/snapshot.py \\
        --customer acme --cluster-identifier portolan-acme-database --region us-east-1 --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

DEFAULT_WAITER_DELAY_SECONDS = 30
DEFAULT_WAITER_MAX_ATTEMPTS = 120  # 30s * 120 = 60 minutes


def snapshot_identifier(customer: str, when: dt.datetime | None = None) -> str:
    """`portolan-<customer>-predeploy-<UTC timestamp>` — sortable, and
    identifies both the customer and why the snapshot exists."""
    when = when or dt.datetime.now(dt.UTC)
    return f"portolan-{customer}-predeploy-{when.strftime('%Y%m%dT%H%M%SZ')}"


def create_and_wait(
    *,
    cluster_identifier: str,
    snapshot_id: str,
    region: str,
    dry_run: bool,
    waiter_delay: int = DEFAULT_WAITER_DELAY_SECONDS,
    waiter_max_attempts: int = DEFAULT_WAITER_MAX_ATTEMPTS,
) -> str:
    """Create `snapshot_id` for `cluster_identifier` and block until it is
    available. Returns the snapshot ARN.

    In `--dry-run`, prints the plan and returns a synthetic ARN without
    calling AWS — no account exists yet for this to run against for real.
    """
    if dry_run:
        timeout_minutes = waiter_delay * waiter_max_attempts // 60
        print(
            f"[dry-run] would create snapshot {snapshot_id!r} of cluster "
            f"{cluster_identifier!r} in {region}, then wait up to "
            f"{timeout_minutes} minutes for it to become available"
        )
        return f"arn:aws:rds:{region}:000000000000:cluster-snapshot:{snapshot_id}"

    import boto3

    client = boto3.client("rds", region_name=region)
    print(f"Creating snapshot {snapshot_id!r} of cluster {cluster_identifier!r}...")
    response = client.create_db_cluster_snapshot(
        DBClusterSnapshotIdentifier=snapshot_id,
        DBClusterIdentifier=cluster_identifier,
    )
    arn = response["DBClusterSnapshot"]["DBClusterSnapshotArn"]

    print(f"Waiting for {snapshot_id!r} to become available (this can take several minutes)...")
    waiter = client.get_waiter("db_cluster_snapshot_available")
    waiter.wait(
        DBClusterSnapshotIdentifier=snapshot_id,
        WaiterConfig={"Delay": waiter_delay, "MaxAttempts": waiter_max_attempts},
    )
    print(f"Snapshot ready: {arn}")
    return arn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--customer", required=True, help="Customer name, as in customers.yaml")
    parser.add_argument(
        "--cluster-identifier",
        required=True,
        help="Aurora cluster identifier (that customer stack's Database resource)",
    )
    parser.add_argument("--region", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    snapshot_id = snapshot_identifier(args.customer)
    try:
        create_and_wait(
            cluster_identifier=args.cluster_identifier,
            snapshot_id=snapshot_id,
            region=args.region,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # top-level CLI guard: report the failure, exit non-zero
        print(f"snapshot failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
