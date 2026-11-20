"""Portolan CDK app entry point.

Usage:
    uv run python infra/app.py                    # --dry (default): placeholder account ids
    uv run python infra/app.py --no-dry            # real account ids, read from the environment

Reads `infra/customers.yaml` (a plain YAML list of `{name, tier, region}`,
see `infra/README.md` "How to add a customer") and instantiates one
`PortolanCustomerStack` per row, plus one shared `PortolanPlatformStack`.

Account ids never live in this repo. In `--dry` mode (the default) every
stack gets a placeholder account id, which is enough for `App().synth()`
and for `aws_cdk.assertions.Template` to inspect the result — exactly what
this wave needs, since there is no AWS account yet. With `--no-dry`, each
account id is read from an environment variable instead:
`PORTOLAN_PLATFORM_ACCOUNT_ID` for the shared platform account, and
`PORTOLAN_ACCOUNT_<NAME>` (the customer's name, upper-cased, non-alphanumeric
characters replaced with `_`) per customer.

This only ever calls `App().synth()` — nothing here deploys anything
(`01-spec/build-contracts.md`, "Infra": "CDK Python app; not deployed until
the AWS account exists").
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from aws_cdk import App, Environment

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portolan_infra.customer_stack import CustomerConfig, PortolanCustomerStack  # noqa: E402
from portolan_infra.platform_stack import PortolanPlatformStack  # noqa: E402
from portolan_infra.tiers import VALID_TIERS  # noqa: E402

DEFAULT_CUSTOMERS_FILE = Path(__file__).resolve().parent / "customers.yaml"
DEFAULT_PLATFORM_REGION = "us-east-1"

# Conventional 12-digit dummy AWS account ids, used only in --dry mode.
PLACEHOLDER_CUSTOMER_ACCOUNT_ID = "111111111111"
PLACEHOLDER_PLATFORM_ACCOUNT_ID = "222222222222"

_REQUIRED_CUSTOMER_KEYS = {"name", "tier", "region"}


@dataclass(frozen=True)
class CustomerSpec:
    """One row of `customers.yaml`, before an account id is attached.

    `domain_name` is optional in the YAML (fix wave 2026-09-12, review
    finding F13): every tier except `"dev"` requires one --
    `PortolanCustomerStack` itself raises if it's missing there, so a
    typo'd or forgotten domain fails at synth, not silently as plaintext
    on the CloudFront-to-ALB hop.
    """

    name: str
    tier: str
    region: str
    domain_name: str | None = None


def load_customers(path: Path) -> list[CustomerSpec]:
    """Parse and validate `customers.yaml`.

    Raises with a message pointing at `infra/README.md` on a missing file,
    a malformed row, or an unknown tier — this file is meant to be edited
    by hand, so failures here should read like a fixable checklist, not a
    stack trace.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See infra/README.md, 'How to add a customer'.")
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        raise ValueError(
            f"{path} must be a YAML list of {{name, tier, region}} entries, "
            f"got {type(raw).__name__}"
        )
    specs: list[CustomerSpec] = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError(
                f"customer entry {row!r} must be a mapping with keys {_REQUIRED_CUSTOMER_KEYS}"
            )
        missing = _REQUIRED_CUSTOMER_KEYS - row.keys()
        if missing:
            raise ValueError(f"customer entry {row!r} missing required key(s): {sorted(missing)}")
        if row["tier"] not in VALID_TIERS:
            raise ValueError(
                f"customer {row['name']!r}: unknown tier {row['tier']!r}, "
                f"expected one of {VALID_TIERS}"
            )
        specs.append(
            CustomerSpec(
                name=row["name"],
                tier=row["tier"],
                region=row["region"],
                domain_name=row.get("domain_name"),
            )
        )
    return specs


def _account_env_var(customer_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", customer_name).strip("_").upper()
    return f"PORTOLAN_ACCOUNT_{slug}"


def resolve_customer_account_id(customer_name: str, *, dry: bool) -> str:
    if dry:
        return PLACEHOLDER_CUSTOMER_ACCOUNT_ID
    env_var = _account_env_var(customer_name)
    account_id = os.environ.get(env_var)
    if not account_id:
        raise RuntimeError(
            f"--no-dry requires {env_var} set to {customer_name}'s AWS account id "
            "(see infra/README.md, 'How to add a customer')."
        )
    return account_id


def resolve_platform_account_id(*, dry: bool) -> str:
    if dry:
        return PLACEHOLDER_PLATFORM_ACCOUNT_ID
    account_id = os.environ.get("PORTOLAN_PLATFORM_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("--no-dry requires PORTOLAN_PLATFORM_ACCOUNT_ID to be set.")
    return account_id


def build_app(
    *,
    dry: bool = True,
    customers_file: Path = DEFAULT_CUSTOMERS_FILE,
    platform_region: str = DEFAULT_PLATFORM_REGION,
) -> App:
    """Construct (but do not synth) the full `App`: one platform stack, one
    customer stack per row of `customers_file`."""
    app = App()

    platform_account_id = resolve_platform_account_id(dry=dry)
    PortolanPlatformStack(
        app,
        "PortolanPlatform",
        env=Environment(account=platform_account_id, region=platform_region),
    )

    # Set by infra/deploy/canary.py and infra/deploy/rollback.py (via
    # `cdk deploy --context` is awkward across accounts; an env var is
    # simpler for a script that already shells out per customer).
    image_tag = os.environ.get("PORTOLAN_IMAGE_TAG", "latest")
    # Review finding F18: PortolanPlatformStack's ECR repo is
    # image_tag_mutability=IMMUTABLE, so a real ("latest") tag can only
    # ever be pushed once -- every deploy after the first would fail at
    # `docker push`, not at synth, which is a much worse place to find
    # out. `--dry` synth (nothing is actually pushed) is exempt; a real
    # deploy must name a real, unique tag (canary.py/rollback.py always
    # pass one -- this only fires for a `--no-dry` run invoked by hand
    # with neither `--image-tag` nor `PORTOLAN_IMAGE_TAG` set).
    if not dry and image_tag == "latest":
        raise RuntimeError(
            "--no-dry requires a real, unique PORTOLAN_IMAGE_TAG (e.g. the commit SHA "
            "the image was built from) -- the shared ECR repository's tags are immutable, "
            "so 'latest' can only ever be pushed once. infra/deploy/canary.py and "
            "rollback.py already pass one; set it yourself if invoking cdk directly."
        )

    for spec in load_customers(customers_file):
        account_id = resolve_customer_account_id(spec.name, dry=dry)
        customer_config = CustomerConfig(
            name=spec.name,
            tier=spec.tier,
            region=spec.region,
            account_id=account_id,
            platform_account_id=platform_account_id,
            image_tag=image_tag,
            domain_name=spec.domain_name,
        )
        PortolanCustomerStack(
            app,
            f"PortolanCustomer-{spec.name}",
            config=customer_config,
            env=Environment(account=account_id, region=spec.region),
        )

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synthesise with placeholder account ids instead of the environment (default: on).",
    )
    parser.add_argument("--customers", type=Path, default=DEFAULT_CUSTOMERS_FILE)
    parser.add_argument("--platform-region", default=DEFAULT_PLATFORM_REGION)
    # parse_known_args: if this is ever invoked as the CDK CLI's `app`
    # command, the CLI appends its own flags (--context, ...) that this
    # parser doesn't need to understand.
    args, _unknown = parser.parse_known_args(argv)

    app = build_app(
        dry=args.dry, customers_file=args.customers, platform_region=args.platform_region
    )
    app.synth()


if __name__ == "__main__":
    main()
