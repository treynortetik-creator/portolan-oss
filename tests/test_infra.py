"""Tests for the Portolan CDK app (`infra/`).

Ownership: `infra/**` and `tests/test_infra*.py` (`01-spec/build-contracts.md`,
"Infra", wave D1). `infra/` is not on the default import path (it isn't part
of the `portolan` package), so this module adds it to `sys.path` itself,
the same way `infra/app.py` does — no other test file needs to know this.

No AWS credentials, network access, or the `cdk` CLI are needed: everything
here is `App().synth()` plus `aws_cdk.assertions.Template`, which works
against the CDK Python libraries alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from aws_cdk import App, Environment
from aws_cdk.assertions import Match, Template

INFRA_DIR = Path(__file__).resolve().parents[1] / "infra"
sys.path.insert(0, str(INFRA_DIR))

from portolan_infra.bedrock_models import (  # noqa: E402
    EXTRACTION_AND_COMPARE_MODEL_IDS,
)
from portolan_infra.customer_stack import CustomerConfig, PortolanCustomerStack  # noqa: E402
from portolan_infra.platform_stack import PortolanPlatformStack  # noqa: E402
from portolan_infra.tiers import TIERS, get_tier  # noqa: E402

PLACEHOLDER_ACCOUNT_ID = "111111111111"
PLACEHOLDER_PLATFORM_ACCOUNT_ID = "222222222222"


def make_customer_config(**overrides) -> CustomerConfig:
    defaults = dict(
        name="acme",
        tier="small",
        region="us-east-1",
        account_id=PLACEHOLDER_ACCOUNT_ID,
        platform_account_id=PLACEHOLDER_PLATFORM_ACCOUNT_ID,
        # Review finding F13: every tier except "dev" now fails synth
        # without a real domain (CloudFront needs an ACM cert on the ALB,
        # and a cert needs a domain) -- a placeholder domain here keeps
        # every other test in this file exercising the real HTTPS path,
        # the same as an actual customer always would.
        domain_name="acme.customers.portolan.example",
    )
    defaults.update(overrides)
    return CustomerConfig(**defaults)


def synth_customer_stack(config: CustomerConfig | None = None) -> Template:
    config = config or make_customer_config()
    app = App()
    stack = PortolanCustomerStack(
        app,
        "TestCustomerStack",
        config=config,
        env=Environment(account=config.account_id, region=config.region),
    )
    return Template.from_stack(stack)


def synth_platform_stack() -> Template:
    app = App()
    stack = PortolanPlatformStack(
        app,
        "TestPlatformStack",
        env=Environment(account=PLACEHOLDER_PLATFORM_ACCOUNT_ID, region="us-east-1"),
    )
    return Template.from_stack(stack)


# -- Customer stack: synthesis --------------------------------------------


def test_customer_stack_synthesises_with_placeholder_ids() -> None:
    template = synth_customer_stack()
    # A real template with real resources came out the other end, not an
    # empty shell — one Aurora cluster, one VPC.
    template.resource_count_is("AWS::RDS::DBCluster", 1)
    template.resource_count_is("AWS::EC2::VPC", 1)


@pytest.mark.parametrize("tier", ["small", "medium", "large"])
def test_customer_stack_synthesises_for_every_tier(tier: str) -> None:
    template = synth_customer_stack(make_customer_config(tier=tier))
    template.resource_count_is("AWS::RDS::DBCluster", 1)


# -- Aurora -----------------------------------------------------------


def test_aurora_engine_version_is_17() -> None:
    template = synth_customer_stack()
    template.has_resource_properties(
        "AWS::RDS::DBCluster",
        {"EngineVersion": Match.string_like_regexp(r"^17\.")},
    )


def test_aurora_deletion_protection_and_encryption() -> None:
    template = synth_customer_stack()
    template.has_resource_properties(
        "AWS::RDS::DBCluster",
        {
            "DeletionProtection": True,
            "StorageEncrypted": True,
            "BackupRetentionPeriod": 35,
        },
    )


def test_aurora_has_no_public_access() -> None:
    template = synth_customer_stack()
    # `publicly_accessible` is set on the writer instance, not the cluster.
    template.has_resource_properties(
        "AWS::RDS::DBInstance",
        {"PubliclyAccessible": False},
    )


# -- S3 originals bucket -----------------------------------------------


def test_originals_bucket_versioned_and_public_access_blocked() -> None:
    template = synth_customer_stack()
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "VersioningConfiguration": {"Status": "Enabled"},
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
        },
    )


def test_originals_bucket_object_lock_is_opt_in() -> None:
    # Off by default...
    default_template = synth_customer_stack()
    default_resources = default_template.find_resources(
        "AWS::S3::Bucket", {"Properties": {"ObjectLockEnabled": True}}
    )
    assert default_resources == {}

    # ...on when the customer's config asks for it.
    locked_template = synth_customer_stack(make_customer_config(enable_object_lock=True))
    locked_resources = locked_template.find_resources(
        "AWS::S3::Bucket", {"Properties": {"ObjectLockEnabled": True}}
    )
    assert len(locked_resources) == 1


# -- SQS: dead-letter queue ----------------------------------------------


def test_dead_letter_queue_has_14_day_retention() -> None:
    template = synth_customer_stack()
    template.has_resource_properties(
        "AWS::SQS::Queue",
        {
            "QueueName": Match.string_like_regexp(r"-ingest-dlq$"),
            "MessageRetentionPeriod": 14 * 24 * 60 * 60,
        },
    )


def test_ingest_queue_points_at_the_dead_letter_queue() -> None:
    template = synth_customer_stack()
    template.has_resource_properties(
        "AWS::SQS::Queue",
        {
            "QueueName": Match.string_like_regexp(r"-ingest$"),
            "RedrivePolicy": Match.object_like({"maxReceiveCount": 5}),
        },
    )


# -- IAM task role: Bedrock model ids ------------------------------------


def test_task_role_policy_references_every_bedrock_model_id() -> None:
    template = synth_customer_stack()
    template_json = template.to_json()
    rendered = str(template_json)
    for model_id in EXTRACTION_AND_COMPARE_MODEL_IDS:
        assert model_id in rendered, f"{model_id!r} not found in any IAM policy resource"


def test_task_role_policy_grants_invoke_model() -> None:
    template = synth_customer_stack()
    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": Match.object_like(
                {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Sid": "BedrockIngestModels",
                                    "Action": Match.array_with(["bedrock:InvokeModel"]),
                                    "Effect": "Allow",
                                }
                            )
                        ]
                    )
                }
            )
        },
    )


# -- Tagging -----------------------------------------------------------

CUSTOMER_TAG = {"Key": "portolan:customer", "Value": "acme"}
TIER_TAG = {"Key": "portolan:tier", "Value": "small"}

# Representative taggable resource types across the stack. "Every resource"
# per the build instructions means every resource CDK can tag — several CFN
# resource types (IAM::Policy, EC2 route/subnet associations, ...) have no
# Tags property at all and are not part of that contract.
TAGGABLE_RESOURCE_TYPES = [
    "AWS::EC2::VPC",
    "AWS::RDS::DBCluster",
    "AWS::S3::Bucket",
    "AWS::SQS::Queue",
    "AWS::KMS::Key",
    "AWS::Logs::LogGroup",
    "AWS::ECS::Cluster",
]


@pytest.mark.parametrize("resource_type", TAGGABLE_RESOURCE_TYPES)
def test_every_taggable_resource_carries_customer_and_tier_tags(resource_type: str) -> None:
    template = synth_customer_stack()
    resources = template.find_resources(resource_type)
    assert resources, f"expected at least one {resource_type} resource"
    for logical_id, resource in resources.items():
        tags = resource["Properties"].get("Tags")
        assert tags is not None, f"{resource_type} {logical_id} has no Tags property"
        assert CUSTOMER_TAG in tags, f"{resource_type} {logical_id} missing {CUSTOMER_TAG}"
        assert TIER_TAG in tags, f"{resource_type} {logical_id} missing {TIER_TAG}"


def test_cognito_user_pool_carries_customer_and_tier_tags() -> None:
    # CfnUserPool renders tags as the free-form `UserPoolTags` map, not the
    # standard `Tags` list every other resource above uses -- Cognito's own
    # CloudFormation shape, not a gap in this stack's tagging.
    template = synth_customer_stack()
    template.has_resource_properties(
        "AWS::Cognito::UserPool",
        {"UserPoolTags": {"portolan:customer": "acme", "portolan:tier": "small"}},
    )


# -- Platform stack ------------------------------------------------------


def test_platform_stack_has_github_oidc_provider() -> None:
    template = synth_platform_stack()
    template.resource_count_is("Custom::AWSCDKOpenIdConnectProvider", 1)


def test_platform_stack_deploy_role_trusts_the_right_repo_and_branch() -> None:
    template = synth_platform_stack()
    rendered = str(template.to_json())
    assert "maintainertetik-creator/portolan-oss" in rendered
    assert "refs/heads/main" in rendered


def test_platform_stack_has_hipaa_conformance_pack() -> None:
    template = synth_platform_stack()
    template.has_resource_properties(
        "AWS::Config::ConformancePack",
        {"ConformancePackName": "Operational-Best-Practices-for-HIPAA-Security"},
    )


def test_platform_stack_security_hub_uses_foundational_standard_not_hipaa() -> None:
    # Security Hub has no HIPAA standard (research-aws-baa-2026-09-12.md
    # §4) -- HIPAA coverage comes from the Config conformance pack above.
    # This checks the standard ARN names "aws-foundational-security-best-practices".
    template = synth_platform_stack()
    template.has_resource_properties(
        "AWS::SecurityHub::Standard",
        {"StandardsArn": Match.string_like_regexp(r"aws-foundational-security-best-practices")},
    )


def test_platform_stack_has_ecr_repo_with_scan_on_push() -> None:
    template = synth_platform_stack()
    template.has_resource_properties(
        "AWS::ECR::Repository",
        {"ImageScanningConfiguration": {"ScanOnPush": True}},
    )


def test_platform_stack_has_organization_cloudtrail() -> None:
    template = synth_platform_stack()
    template.has_resource_properties(
        "AWS::CloudTrail::Trail",
        {"IsOrganizationTrail": True, "IsMultiRegionTrail": True},
    )


def test_platform_stack_has_guardduty_detector() -> None:
    template = synth_platform_stack()
    template.has_resource_properties("AWS::GuardDuty::Detector", {"Enable": True})


# -- Edge: CloudFront/ALB TLS (review finding F13) -----------------------


def test_dev_tier_without_domain_stays_http_to_the_alb() -> None:
    template = synth_customer_stack(make_customer_config(tier="dev", domain_name=None))
    template.has_resource_properties(
        "AWS::CloudFront::Distribution",
        {
            "DistributionConfig": Match.object_like(
                {
                    "Origins": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "CustomOriginConfig": Match.object_like(
                                        {"OriginProtocolPolicy": "http-only"}
                                    )
                                }
                            )
                        ]
                    )
                }
            )
        },
    )
    template.resource_count_is("AWS::CertificateManager::Certificate", 0)


@pytest.mark.parametrize("tier", ["small", "medium", "large"])
def test_non_dev_tier_without_domain_fails_synth(tier: str) -> None:
    with pytest.raises(ValueError, match="domain_name is required"):
        synth_customer_stack(make_customer_config(tier=tier, domain_name=None))


def test_a_real_domain_gets_an_acm_certificate_and_https_origin() -> None:
    template = synth_customer_stack()  # make_customer_config's default domain_name
    template.resource_count_is("AWS::CertificateManager::Certificate", 1)
    template.has_resource_properties(
        "AWS::CloudFront::Distribution",
        {
            "DistributionConfig": Match.object_like(
                {
                    "Origins": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "CustomOriginConfig": Match.object_like(
                                        {"OriginProtocolPolicy": "https-only"}
                                    )
                                }
                            )
                        ]
                    )
                }
            )
        },
    )


def test_web_task_receives_auth_secrets_and_bedrock_provider() -> None:
    template = synth_customer_stack()
    template.resource_count_is("AWS::SecretsManager::Secret", 4)  # DB + 3 auth secrets
    rendered = str(template.to_json())
    for name in ("PORTOLAN_ORG_KEY", "PORTOLAN_SESSION_SECRET", "PORTOLAN_MCP_ORG_KEY"):
        assert name in rendered
    template.has_resource_properties(
        "AWS::ECS::TaskDefinition",
        {
            "ContainerDefinitions": Match.array_with(
                [
                    Match.object_like(
                        {
                            "Environment": Match.array_with(
                                [{"Name": "LLM_PROVIDER", "Value": "bedrock"}]
                            )
                        }
                    )
                ]
            )
        },
    )


def test_task_role_granted_exactly_once() -> None:
    # Review finding F13 (slop list): _grant_task_resources used to run
    # once per service against the same shared task_role, doubling every
    # IAM statement. Both services' containers are on one shared role, so
    # each policy statement this grants should appear exactly once.
    template = synth_customer_stack()
    rendered = template.to_json()
    ingest_queue_grants = 0
    for policy in rendered["Resources"].values():
        if policy["Type"] != "AWS::IAM::Policy":
            continue
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = statement.get("Action")
            actions = actions if isinstance(actions, list) else [actions]
            if any(a in ("sqs:SendMessage", "sqs:ReceiveMessage") for a in actions):
                ingest_queue_grants += 1
    # grant_send_messages + grant_consume_messages, once each, not twice.
    assert ingest_queue_grants <= 2


# -- Tiers ---------------------------------------------------------------


def test_tiers_match_the_cost_analysis_acu_ranges() -> None:
    # 01-spec/cost-analysis-2026-09-12.md §2, assumption 18.
    assert TIERS["small"].aurora_min_acu == 0.5
    assert TIERS["small"].aurora_max_acu == 2.0
    assert TIERS["medium"].aurora_min_acu == 1.0
    assert TIERS["medium"].aurora_max_acu == 4.0
    assert TIERS["large"].aurora_min_acu == 2.0
    assert TIERS["large"].aurora_max_acu == 8.0


def test_tiers_are_strictly_increasing() -> None:
    small, medium, large = TIERS["small"], TIERS["medium"], TIERS["large"]
    assert small.aurora_max_acu < medium.aurora_max_acu < large.aurora_max_acu
    assert small.worker.cpu <= medium.worker.cpu <= large.worker.cpu
    assert small.worker_max_count < medium.worker_max_count < large.worker_max_count


def test_get_tier_unknown_name_raises_with_valid_options_listed() -> None:
    with pytest.raises(
        ValueError, match=r"small.*medium.*large|large.*medium.*small|small|medium|large"
    ):
        get_tier("extra-large")


def test_customer_stack_uses_tier_sizing_for_aurora_capacity() -> None:
    for tier_name in ["small", "medium", "large"]:
        template = synth_customer_stack(make_customer_config(tier=tier_name))
        sizing = get_tier(tier_name)
        template.has_resource_properties(
            "AWS::RDS::DBCluster",
            {
                "ServerlessV2ScalingConfiguration": {
                    "MinCapacity": sizing.aurora_min_acu,
                    "MaxCapacity": sizing.aurora_max_acu,
                }
            },
        )
