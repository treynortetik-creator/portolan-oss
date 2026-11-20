"""The shared platform account: `PortolanPlatformStack`.

Ownership: `infra/**` (build-contracts.md, "Infra", wave D1). Not deployed —
there is no AWS account, let alone an AWS Organization, yet. This stack
targets the Organization's management (or a delegated security-tooling)
account, per `01-spec/research/research-aws-baa-2026-09-12.md` §8's
week-one checklist: accept the BAA at the org level first, then turn on
CloudTrail/GuardDuty/Config here before any customer account holds PHI.

Security Hub note (research doc §4, correcting an earlier assumption in
the build instructions): Security Hub has **no HIPAA standard**. Its
built-in standards are AWS Foundational Security Best Practices, AI
Security Best Practices, Resource Tagging, CIS, NIST 800-53, NIST 800-171
and PCI DSS. HIPAA-specific checks come from the AWS Config conformance
pack (`Operational-Best-Practices-for-HIPAA-Security`) instead — this
stack enables both: Config's HIPAA pack for the HIPAA-mapped controls, and
Security Hub's **AWS Foundational** standard for general posture, exactly
as the build instructions ask for.
"""

from __future__ import annotations

from aws_cdk import CfnJson, Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudtrail as cloudtrail
from aws_cdk import aws_config as config
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_guardduty as guardduty
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_securityhub as securityhub
from constructs import Construct

GITHUB_OIDC_URL = "https://token.actions.githubusercontent.com"
# GitHub's documented OIDC root CA thumbprint
# (https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services,
# fetched historically; AWS's IAM OIDC provider has not required exact
# thumbprint validation for well-known IdPs since 2023, but the field is
# still mandatory on the CDK construct). Re-confirm before first deploy.
GITHUB_OIDC_THUMBPRINT = "6938fd4d98bab03faadb97b34396831e3780aea1"
DEFAULT_GITHUB_REPO = "maintainertetik-creator/portolan-oss"
DEFAULT_GITHUB_BRANCH = "main"

# AWS's own SSM-document name for the HIPAA sample conformance pack, per the
# naming convention AWS uses for every conformance-pack sample in the
# console's "Deploy conformance pack" catalog. Not independently re-fetched
# from a live AWS page this session — re-confirm the exact document name in
# the Config console before first deploy (per the project's "verify before
# claiming" rule).
HIPAA_CONFORMANCE_PACK_SSM_DOCUMENT = "AWS-Operational-Best-Practices-for-HIPAA-Security"

LOG_RETENTION = logs.RetentionDays.ONE_YEAR


class PortolanPlatformStack(Stack):
    """ECR, GitHub OIDC deploy role, CloudTrail org trail, GuardDuty, Config
    HIPAA conformance pack, Security Hub (AWS Foundational standard)."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        github_repo: str = DEFAULT_GITHUB_REPO,
        github_branch: str = DEFAULT_GITHUB_BRANCH,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.github_repo = github_repo
        self.github_branch = github_branch

        self.repository = self._build_ecr_repository()
        self.oidc_provider, self.deploy_role = self._build_github_oidc()
        self.trail_bucket, self.trail = self._build_cloudtrail()
        self.guardduty_detector = self._build_guardduty()
        self.config_bucket, self.config_recorder, self.conformance_pack = (
            self._build_config_hipaa_pack()
        )
        self.security_hub, self.foundational_standard = self._build_security_hub()

    # -- ECR -----------------------------------------------------------

    def _build_ecr_repository(self) -> ecr.Repository:
        return ecr.Repository(
            self,
            "Repository",
            repository_name="portolan",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            removal_policy=RemovalPolicy.RETAIN,
        )

    # -- GitHub OIDC deploy role --------------------------------------------

    def _build_github_oidc(self) -> tuple[iam.OpenIdConnectProvider, iam.Role]:
        # `aws-cdk-github-oidc` (the community L2 that wraps this pattern)
        # is not in aws-cdk-lib — see infra/README.md's "construct not
        # found" note. Built here from core CDK primitives instead:
        # `iam.OpenIdConnectProvider` + a role whose trust policy is a
        # `FederatedPrincipal` scoped to this one repo and branch via the
        # provider's `sub` claim.
        provider = iam.OpenIdConnectProvider(
            self,
            "GithubOidcProvider",
            url=GITHUB_OIDC_URL,
            client_ids=["sts.amazonaws.com"],
            thumbprints=[GITHUB_OIDC_THUMBPRINT],
        )
        # The claim keys (`<issuer>:aud`, `<issuer>:sub`) embed the
        # provider's issuer, which is only known as a CloudFormation token
        # at synth time (it's derived from the OIDC provider's generated
        # ARN). A plain Python dict can't use a token as a *key* — IAM
        # condition keys resolve at deploy time, not synth time — so each
        # condition's inner map goes through `CfnJson`, which defers key
        # resolution to CloudFormation instead of requiring a literal
        # string now.
        aud_condition = CfnJson(
            self,
            "GithubOidcAudCondition",
            value={f"{provider.open_id_connect_provider_issuer}:aud": "sts.amazonaws.com"},
        )
        sub_condition = CfnJson(
            self,
            "GithubOidcSubCondition",
            value={
                f"{provider.open_id_connect_provider_issuer}:sub": (
                    f"repo:{self.github_repo}:ref:refs/heads/{self.github_branch}"
                )
            },
        )
        role = iam.Role(
            self,
            "GithubDeployRole",
            role_name="portolan-github-deploy",
            description=f"CI deploy role for {self.github_repo}@{self.github_branch}",
            assumed_by=iam.FederatedPrincipal(
                provider.open_id_connect_provider_arn,
                conditions={"StringEquals": aud_condition, "StringLike": sub_condition},
                assume_role_action="sts:AssumeRoleWithWebIdentity",
            ),
            max_session_duration=Duration.hours(1),
        )
        # Deploy permissions are deliberately not attached yet: this role
        # will need `cdk deploy`-shaped access (CloudFormation, the
        # per-service permissions each customer stack touches, and
        # `iam:PassRole` on the roles it creates) once real customer
        # accounts exist to scope those resources to. Attaching
        # AdministratorAccess now, before there is a second AWS account to
        # bound it to, is the wrong default for a HIPAA-adjacent org — left
        # for the deploy-safety wave (spec §11) to scope per-account.
        return provider, role

    # -- CloudTrail -----------------------------------------------------

    def _build_cloudtrail(self) -> tuple[s3.Bucket, cloudtrail.Trail]:
        # "Locked bucket": versioned + Object Lock in governance mode, so a
        # compromised deploy role can't quietly rewrite the audit trail.
        bucket = s3.Bucket(
            self,
            "TrailBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            object_lock_enabled=True,
            object_lock_default_retention=s3.ObjectLockRetention.governance(Duration.days(365)),
            removal_policy=RemovalPolicy.RETAIN,
        )
        trail = cloudtrail.Trail(
            self,
            "OrganizationTrail",
            bucket=bucket,
            is_organization_trail=True,
            is_multi_region_trail=True,
            include_global_service_events=True,
            enable_file_validation=True,
            send_to_cloud_watch_logs=True,
            cloud_watch_logs_retention=LOG_RETENTION,
        )
        return bucket, trail

    # -- GuardDuty -----------------------------------------------------

    def _build_guardduty(self) -> guardduty.CfnDetector:
        return guardduty.CfnDetector(
            self,
            "GuardDutyDetector",
            enable=True,
            finding_publishing_frequency="FIFTEEN_MINUTES",
        )

    # -- Config: HIPAA conformance pack -----------------------------------

    def _build_config_hipaa_pack(
        self,
    ) -> tuple[s3.Bucket, config.CfnConfigurationRecorder, config.CfnConformancePack]:
        bucket = s3.Bucket(
            self,
            "ConfigBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        recorder_role = iam.Role(
            self,
            "ConfigRecorderRole",
            assumed_by=iam.ServicePrincipal("config.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWS_ConfigRole")
            ],
        )
        bucket.grant_read_write(recorder_role)

        recorder = config.CfnConfigurationRecorder(
            self,
            "ConfigRecorder",
            role_arn=recorder_role.role_arn,
            recording_group=config.CfnConfigurationRecorder.RecordingGroupProperty(
                all_supported=True, include_global_resource_types=True
            ),
        )
        delivery_channel = config.CfnDeliveryChannel(
            self, "ConfigDeliveryChannel", s3_bucket_name=bucket.bucket_name
        )

        conformance_pack = config.CfnConformancePack(
            self,
            "HipaaConformancePack",
            conformance_pack_name="Operational-Best-Practices-for-HIPAA-Security",
            delivery_s3_bucket=bucket.bucket_name,
        )
        # Set via a raw property override rather than the typed
        # `template_ssm_document_details=TemplateSSMDocumentDetailsProperty(...)`
        # kwarg: aws-cdk-lib 2.269.0 synthesizes that typed property's
        # `document_name` field as lower-camel `documentName`, which
        # `cdk synth`'s own CloudFormation-spec validator flags as invalid
        # ("Did you mean 'DocumentName'?") — a real property-casing bug in
        # this CDK version for this specific nested type, confirmed by
        # checking the synthesized template against that validator. The
        # override below writes the correctly-cased `DocumentName` key
        # directly, bypassing the buggy marshalling.
        conformance_pack.add_property_override(
            "TemplateSSMDocumentDetails",
            {"DocumentName": HIPAA_CONFORMANCE_PACK_SSM_DOCUMENT},
        )
        conformance_pack.node.add_dependency(recorder)
        conformance_pack.node.add_dependency(delivery_channel)
        return bucket, recorder, conformance_pack

    # -- Security Hub: AWS Foundational standard --------------------------

    def _build_security_hub(self) -> tuple[securityhub.CfnHub, securityhub.CfnStandard]:
        # `enable_default_standards=False`: we want exactly the AWS
        # Foundational standard the build instructions ask for, not
        # whatever Security Hub currently treats as "default" (which has
        # changed over time and, per the research doc, never included a
        # HIPAA option to begin with).
        hub = securityhub.CfnHub(self, "SecurityHub", enable_default_standards=False)
        foundational_standard = securityhub.CfnStandard(
            self,
            "AwsFoundationalStandard",
            standards_arn=(
                f"arn:aws:securityhub:{self.region}::standards/"
                "aws-foundational-security-best-practices/v/1.0.0"
            ),
        )
        foundational_standard.node.add_dependency(hub)
        return hub, foundational_standard
