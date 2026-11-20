"""Deploy order: demo tenant, then a canary customer, then the rest.

Spec: `01-spec/portolan-spec-v2.md` §11 — "Deploy to the demo tenant first,
then a canary customer, then the rest. Post-deploy smoke checks... Automatic
rollback on a failed check." This script sequences that: for each customer,
in order, it snapshots that customer's database (`snapshot.py`), deploys
the CDK stack, then runs the smoke check (`smoke.py`) before moving to the
next customer. A failed smoke check stops the rollout — see
`rollback.py` for redeploying the previous image tag on that customer.

Reads `infra/customers.yaml`. `demo` (by name) always deploys first when
present. The canary is the first remaining customer in file order, unless
`--canary NAME` overrides it. Everyone else follows, in file order.

Per customer, this script looks for two optional environment variables
(named the same way `infra/app.py` names `PORTOLAN_ACCOUNT_<NAME>`) to know
where that customer's database is: `PORTOLAN_DATABASE_URL_<NAME>` and
`PORTOLAN_DB_CLUSTER_<NAME>`. Without them, the snapshot and smoke-check
steps are skipped for that customer with a warning (still true today: there
is no AWS account, so nothing has a database yet) — the deploy step itself
still runs.

Usage:
    uv run python infra/deploy/canary.py --image-tag sha-abc123 --dry-run
    uv run python infra/deploy/canary.py --image-tag sha-abc123
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_INFRA_DIR = Path(__file__).resolve().parents[1]
_DEPLOY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_INFRA_DIR))
# Also this file's own directory, so `import rollback` below (in
# main()'s automatic-rollback call, review finding F14) resolves
# regardless of how this module itself was loaded -- running it as a
# script already gets this for free, but importing it as a module (a
# test, or rollback.py's own `from canary import deploy_stack`) does not.
sys.path.insert(0, str(_DEPLOY_DIR))

from app import DEFAULT_CUSTOMERS_FILE, CustomerSpec, load_customers  # noqa: E402

DEMO_CUSTOMER_NAME = "demo"
SNAPSHOT_SCRIPT = Path(__file__).resolve().parent / "snapshot.py"
SMOKE_SCRIPT = Path(__file__).resolve().parent / "smoke.py"


@dataclass(frozen=True)
class StageResult:
    customer: str
    deployed: bool
    smoke_passed: bool | None  # None: smoke check was skipped (no database env vars set)


def deploy_order(
    customers: list[CustomerSpec], *, canary_override: str | None = None
) -> list[CustomerSpec]:
    """Demo first (if present), then the canary, then everyone else — each
    customer exactly once, in this order."""
    by_name = {c.name: c for c in customers}
    remaining = list(customers)
    ordered: list[CustomerSpec] = []

    def take(name: str) -> None:
        nonlocal remaining
        if name in by_name and by_name[name] in remaining:
            ordered.append(by_name[name])
            remaining = [c for c in remaining if c.name != name]

    take(DEMO_CUSTOMER_NAME)

    canary_name = canary_override or (remaining[0].name if remaining else None)
    if canary_name:
        take(canary_name)

    ordered.extend(remaining)
    return ordered


def _env_var_for(customer_name: str, prefix: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", customer_name).strip("_").upper()
    return f"{prefix}_{slug}"


def _redact_cmd(cmd: list[str]) -> list[str]:
    """`cmd`, with the value following any `--database-url` flag replaced
    for display -- the real value still reaches `subprocess.run`
    unredacted; only what gets printed changes.

    Review finding F14: this used to print the full command including
    `--database-url <url>` verbatim -- the Aurora password lives in that
    URL, and this line goes straight into CI logs.
    """
    redacted = list(cmd)
    for i, token in enumerate(redacted):
        if token == "--database-url" and i + 1 < len(redacted):
            redacted[i + 1] = "***REDACTED***"
    return redacted


def _run(cmd: list[str]) -> int:
    print("$ " + " ".join(_redact_cmd(cmd)))
    return subprocess.run(cmd, check=False).returncode


def deploy_stack(customer_name: str, *, image_tag: str, dry_run: bool) -> int:
    """Deploy one customer's CDK stack via the `cdk` CLI. Assumes AWS
    credentials for that customer's account are already active in the
    environment — this script does not manage credentials.

    In `--dry-run`, prints the command instead of running it and returns 0
    without any network or AWS call — same contract as `snapshot.py` and
    `smoke.py`'s own `--dry-run`, and consistent with this wave's "do not
    deploy anything, there is no AWS account yet."
    """
    stack_name = f"PortolanCustomer-{customer_name}"
    cmd = [
        "npx",
        "--yes",
        "aws-cdk",
        "deploy",
        stack_name,
        "--app",
        "uv run python infra/app.py --no-dry",
        "--exclusively",
        "--require-approval=never",
    ]
    print("$ " + " ".join(cmd) + f"   (PORTOLAN_IMAGE_TAG={image_tag})")
    if dry_run:
        print(f"[dry-run] would run the above for {stack_name}")
        return 0
    env = {**os.environ, "PORTOLAN_IMAGE_TAG": image_tag}
    return subprocess.run(cmd, check=False, env=env).returncode


def run_stage(customer: CustomerSpec, *, image_tag: str, dry_run: bool) -> StageResult:
    print(f"\n=== {customer.name} ({customer.tier}, {customer.region}) ===")

    database_url = os.environ.get(_env_var_for(customer.name, "PORTOLAN_DATABASE_URL"))
    cluster_id = os.environ.get(_env_var_for(customer.name, "PORTOLAN_DB_CLUSTER"))
    baseline_file = _INFRA_DIR / f".smoke-baseline-{customer.name}.json"

    if cluster_id:
        _run(
            [
                "uv",
                "run",
                "python",
                str(SNAPSHOT_SCRIPT),
                "--customer",
                customer.name,
                "--cluster-identifier",
                cluster_id,
                "--region",
                customer.region,
                *(["--dry-run"] if dry_run else []),
            ]
        )
    else:
        print(f"  (no {_env_var_for(customer.name, 'PORTOLAN_DB_CLUSTER')} set: skipping snapshot)")

    if database_url:
        _run(
            [
                "uv",
                "run",
                "python",
                str(SMOKE_SCRIPT),
                "--database-url",
                database_url,
                *(["--dry-run"] if dry_run else []),
                "baseline",
                "--out",
                str(baseline_file),
            ]
        )

    deploy_returncode = deploy_stack(customer.name, image_tag=image_tag, dry_run=dry_run)
    if deploy_returncode != 0:
        return StageResult(customer=customer.name, deployed=False, smoke_passed=None)

    if not database_url:
        env_var = _env_var_for(customer.name, "PORTOLAN_DATABASE_URL")
        print(f"  (no {env_var} set: skipping smoke check)")
        return StageResult(customer=customer.name, deployed=True, smoke_passed=None)

    smoke_returncode = _run(
        [
            "uv",
            "run",
            "python",
            str(SMOKE_SCRIPT),
            "--database-url",
            database_url,
            *(["--dry-run"] if dry_run else []),
            "check",
            "--baseline",
            str(baseline_file),
        ]
    )
    return StageResult(customer=customer.name, deployed=True, smoke_passed=smoke_returncode == 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--customers", type=Path, default=DEFAULT_CUSTOMERS_FILE)
    parser.add_argument(
        "--canary", default=None, help="Override which non-demo customer deploys second"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    customers = load_customers(args.customers)
    order = deploy_order(customers, canary_override=args.canary)
    print("Deploy order: " + " -> ".join(c.name for c in order))

    for customer in order:
        result = run_stage(customer, image_tag=args.image_tag, dry_run=args.dry_run)
        if not result.deployed:
            print(f"\nDEPLOY FAILED at {customer.name}. Stopping rollout.", file=sys.stderr)
            return 1
        if result.smoke_passed is False:
            # Review finding F14: spec sec 11 says "Automatic rollback on
            # a failed check" -- this used to print a pointer at
            # rollback.py and stop, requiring a human to run it by hand.
            print(
                f"\nSMOKE CHECK FAILED at {customer.name}. Rolling back automatically.",
                file=sys.stderr,
            )
            import rollback as rollback_module

            rollback_returncode = rollback_module.main(
                [
                    "--customer",
                    customer.name,
                    "--bad-tag",
                    args.image_tag,
                    *(["--dry-run"] if args.dry_run else []),
                ]
            )
            if rollback_returncode != 0:
                print(
                    f"\nROLLBACK ALSO FAILED for {customer.name} -- this needs a human "
                    "now. See infra/deploy/rollback.py.",
                    file=sys.stderr,
                )
            print(f"\nStopping rollout at {customer.name}.", file=sys.stderr)
            return 1

    print("\nRollout complete: " + ", ".join(c.name for c in order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
