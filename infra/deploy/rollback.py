"""Redeploy the previous image tag for one customer.

Spec: `01-spec/portolan-spec-v2.md` §11 — "Automatic rollback on a failed
check." Called (by hand, or by CI) after `canary.py` reports a failed
smoke check: finds the tag that was deployed immediately before
`--bad-tag` in the shared ECR repository's push history, and redeploys
that customer's stack pinned to it. `--to-tag` skips the ECR lookup and
rolls back to an exact, known-good tag instead.

Usage:
    uv run python infra/deploy/rollback.py --customer acme --bad-tag sha-abc123 --dry-run
    uv run python infra/deploy/rollback.py --customer acme --bad-tag sha-abc123
    uv run python infra/deploy/rollback.py --customer acme --to-tag sha-known-good
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from canary import deploy_stack  # noqa: E402


def previous_image_tag(repository_name: str, bad_tag: str, *, region: str) -> str:
    """The tag pushed immediately before `bad_tag`, by push time, in the
    named ECR repository.

    Raises `ValueError` if `bad_tag` isn't found, or is the oldest tag (no
    "previous" to roll back to) — both are fixable-by-a-human situations
    (typo in `--bad-tag`, or this is the very first release), not a crash.
    """
    import boto3

    client = boto3.client("ecr", region_name=region)
    paginator = client.get_paginator("describe_images")
    images: list[dict[str, Any]] = []
    for page in paginator.paginate(repositoryName=repository_name):
        images.extend(page["imageDetails"])
    images.sort(key=lambda image: image["imagePushedAt"])  # oldest first

    tags_in_push_order: list[str] = []
    for image in images:
        tags_in_push_order.extend(image.get("imageTags", []))

    try:
        bad_index = tags_in_push_order.index(bad_tag)
    except ValueError as exc:
        raise ValueError(
            f"tag {bad_tag!r} not found in {repository_name!r}'s push history"
        ) from exc
    if bad_index == 0:
        raise ValueError(
            f"tag {bad_tag!r} is the oldest tag in {repository_name!r}; nothing to roll back to"
        )
    return tags_in_push_order[bad_index - 1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--customer", required=True)
    parser.add_argument("--repository-name", default="portolan")
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="Region the shared ECR repository lives in (the platform account's region)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bad-tag", help="The tag that failed smoke; roll back to whatever preceded it"
    )
    group.add_argument("--to-tag", help="Skip the ECR lookup; roll back to this exact tag")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.to_tag:
        target_tag = args.to_tag
    elif args.dry_run:
        target_tag = "<previous-tag>"
        print(
            f"[dry-run] would look up the tag pushed before {args.bad_tag!r} in ECR repo "
            f"{args.repository_name!r} ({args.region})"
        )
    else:
        try:
            target_tag = previous_image_tag(args.repository_name, args.bad_tag, region=args.region)
        except ValueError as exc:
            print(f"rollback failed: {exc}", file=sys.stderr)
            return 1
        print(f"Rolling {args.customer} back from {args.bad_tag!r} to {target_tag!r}")

    returncode = deploy_stack(args.customer, image_tag=target_tag, dry_run=args.dry_run)
    return 0 if returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
