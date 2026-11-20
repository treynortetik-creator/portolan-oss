"""One single-tenant customer environment: `PortolanCustomerStack`.

Deployed once per customer into that customer's own AWS account (spec §11:
"Our AWS account. One AWS account per customer under our Organization:
isolated data, shared code"). Every resource below is tagged
`portolan:customer` / `portolan:tier` per the build instructions, so a cost
or security report can be sliced per customer even though each customer
already has a whole account to themselves.

Ownership: `infra/**` (build-contracts.md, "Infra", wave D1). Not deployed —
there is no AWS account yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack, Tags
from aws_cdk import aws_applicationautoscaling as appscaling
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_rds as rds
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_wafv2 as wafv2
from constructs import Construct

from portolan_infra.bedrock_models import bedrock_model_arns
from portolan_infra.tiers import TierSizing, get_tier

AURORA_ENGINE_VERSION = rds.AuroraPostgresEngineVersion.VER_17_9
BACKUP_RETENTION = Duration.days(35)
DLQ_RETENTION = Duration.days(14)
LOG_RETENTION = logs.RetentionDays.ONE_YEAR
S3_INTELLIGENT_TIERING_AFTER = Duration.days(90)


@dataclass(frozen=True)
class CustomerConfig:
    """One row of `customers.yaml`, validated and typed."""

    name: str
    tier: str
    region: str
    account_id: str
    platform_account_id: str
    github_repo: str = "maintainertetik-creator/portolan-oss"
    image_tag: str = "latest"
    enable_object_lock: bool = False
    # Review finding F13: CloudFront cannot present real HTTPS to the ALB
    # without an ACM certificate, and a certificate needs a real domain.
    # `None` is only tolerated for `tier == "dev"` (`_build_alb_certificate`
    # raises for every other tier) -- a real customer environment needs a
    # real domain before it deploys, full stop.
    domain_name: str | None = None


class PortolanCustomerStack(Stack):
    """VPC, Aurora, S3, SQS, ECS/Fargate, CloudFront+WAF, Cognito, IAM, logs.

    One instance of this stack is a complete, isolated customer
    environment. `config.tier` selects sizing from
    `portolan_infra.tiers.TIERS` (sourced from
    `01-spec/cost-analysis-2026-09-12.md`).
    """

    def __init__(
        self, scope: Construct, construct_id: str, *, config: CustomerConfig, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.config = config
        sizing = get_tier(config.tier)

        Tags.of(self).add("portolan:customer", config.name)
        Tags.of(self).add("portolan:tier", config.tier)

        self.kms_key = self._build_kms_key()
        self.vpc = self._build_vpc()
        self.database = self._build_database(sizing)
        self.originals_bucket, self._access_logs_bucket = self._build_originals_bucket(config)
        self.dead_letter_queue, self.ingest_queue = self._build_queues()
        self.log_group_web, self.log_group_worker = self._build_log_groups()
        self.task_role = self._build_task_role()
        # Review finding F13 (slop list): this used to run once per
        # service (_build_web_service, _build_worker_service) against
        # what is, in both cases, this same self.task_role -- doubling
        # every IAM policy statement on it. Granted exactly once, here.
        self._grant_task_resources(self.task_role)
        self.auth_secrets = self._build_auth_secrets()
        self.cluster = ecs.Cluster(
            self, "Cluster", vpc=self.vpc, container_insights_v2=ecs.ContainerInsights.ENABLED
        )
        self.repository = ecr.Repository.from_repository_attributes(
            self,
            "SharedRepository",
            repository_arn=(
                f"arn:aws:ecr:{self.region}:{config.platform_account_id}:repository/portolan"
            ),
            repository_name="portolan",
        )
        self._alb_certificate = self._build_alb_certificate()
        self.web_service = self._build_web_service(sizing)
        self.worker_service = self._build_worker_service(sizing)
        self.web_acl = self._build_web_acl()
        self.distribution = self._build_distribution()
        self.user_pool, self.user_pool_client = self._build_cognito()

        self._outputs()

    # -- KMS -----------------------------------------------------------

    def _build_kms_key(self) -> kms.Key:
        return kms.Key(
            self,
            "DataKey",
            alias=f"portolan-{self.config.name}",
            description=f"Portolan customer data key for {self.config.name}",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

    # -- Network ---------------------------------------------------------

    def _build_vpc(self) -> ec2.Vpc:
        # Two AZs, private subnets, one NAT gateway — per the build
        # instructions. A public subnet is still required to host the NAT
        # gateway itself; the ALB is internal (see _build_web_service) so
        # nothing customer-facing lives in it. Aurora sits in the isolated
        # tier (no route to the internet at all, in or out).
        return ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="app", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="data", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=24
                ),
            ],
        )

    # -- Database ----------------------------------------------------------

    def _build_database(self, sizing: TierSizing) -> rds.DatabaseCluster:
        return rds.DatabaseCluster(
            self,
            "Database",
            engine=rds.DatabaseClusterEngine.aurora_postgres(version=AURORA_ENGINE_VERSION),
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED),
            writer=rds.ClusterInstance.serverless_v2("Writer", publicly_accessible=False),
            serverless_v2_min_capacity=sizing.aurora_min_acu,
            serverless_v2_max_capacity=sizing.aurora_max_acu,
            default_database_name="portolan",
            credentials=rds.Credentials.from_generated_secret("portolan_admin"),
            storage_encrypted=True,
            storage_encryption_key=self.kms_key,
            deletion_protection=True,
            backup=rds.BackupProps(retention=BACKUP_RETENTION),
            cloudwatch_logs_exports=["postgresql"],
            cloudwatch_logs_retention=LOG_RETENTION,
            removal_policy=RemovalPolicy.RETAIN,
        )

    # -- Storage ----------------------------------------------------------

    def _build_originals_bucket(self, config: CustomerConfig) -> tuple[s3.Bucket, s3.Bucket]:
        # S3 server access logging cannot write into an SSE-KMS bucket, so
        # the log destination is a separate, SSE-S3 bucket.
        access_logs_bucket = s3.Bucket(
            self,
            "OriginalsAccessLogsBucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        object_lock_kwargs: dict = {}
        if config.enable_object_lock:
            object_lock_kwargs = {
                "object_lock_enabled": True,
                # Governance mode per the build instructions ("object lock
                # in governance mode optional flag") — a year, matching the
                # backup retention window; adjust per customer contract.
                "object_lock_default_retention": s3.ObjectLockRetention.governance(
                    Duration.days(365)
                ),
            }
        originals_bucket = s3.Bucket(
            self,
            "OriginalsBucket",
            versioned=True,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.kms_key,
            bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            server_access_logs_bucket=access_logs_bucket,
            server_access_logs_prefix="originals-access-logs/",
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="intelligent-tiering-after-90-days",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                            transition_after=S3_INTELLIGENT_TIERING_AFTER,
                        )
                    ],
                )
            ],
            removal_policy=RemovalPolicy.RETAIN,
            **object_lock_kwargs,
        )
        return originals_bucket, access_logs_bucket

    # -- Queues -------------------------------------------------------------

    def _build_queues(self) -> tuple[sqs.Queue, sqs.Queue]:
        dead_letter_queue = sqs.Queue(
            self,
            "IngestDeadLetterQueue",
            queue_name=f"portolan-{self.config.name}-ingest-dlq",
            retention_period=DLQ_RETENTION,
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=self.kms_key,
            enforce_ssl=True,
        )
        ingest_queue = sqs.Queue(
            self,
            "IngestQueue",
            queue_name=f"portolan-{self.config.name}-ingest",
            visibility_timeout=Duration.minutes(5),
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=self.kms_key,
            enforce_ssl=True,
            dead_letter_queue=sqs.DeadLetterQueue(max_receive_count=5, queue=dead_letter_queue),
        )
        return dead_letter_queue, ingest_queue

    # -- Logs -----------------------------------------------------------

    def _build_log_groups(self) -> tuple[logs.LogGroup, logs.LogGroup]:
        web = logs.LogGroup(
            self,
            "WebLogGroup",
            log_group_name=f"/portolan/{self.config.name}/web",
            retention=LOG_RETENTION,
            removal_policy=RemovalPolicy.RETAIN,
        )
        worker = logs.LogGroup(
            self,
            "WorkerLogGroup",
            log_group_name=f"/portolan/{self.config.name}/worker",
            retention=LOG_RETENTION,
            removal_policy=RemovalPolicy.RETAIN,
        )
        return web, worker

    # -- IAM -----------------------------------------------------------

    def _build_task_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "TaskRole",
            role_name=f"portolan-{self.config.name}-task",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description=(
                f"Portolan {self.config.name} ECS task role: Bedrock ingest models, "
                "originals bucket, ingest queue, DB credentials secret."
            ),
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockIngestModels",
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=bedrock_model_arns(self.region, self.account),
            )
        )
        return role

    # -- Auth secrets -------------------------------------------------------

    def _build_auth_secrets(self) -> dict[str, secretsmanager.Secret]:
        """`PORTOLAN_ORG_KEY`, `PORTOLAN_SESSION_SECRET`,
        `PORTOLAN_MCP_ORG_KEY` (review finding F1): the web and worker
        tasks used to get neither an org key nor a session secret at all,
        so a deployed Portolan ran with `portolan.web.auth`'s old
        hardcoded fallback -- the review forged a cookie from that exact
        literal and read the whole graph back. Each is a real, stack-
        generated random secret; nothing here is a customer-supplied or
        hardcoded value.
        """
        names = {
            "org_key": "org-key",
            "session_secret": "session-secret",
            "mcp_org_key": "mcp-org-key",
        }
        return {
            key: secretsmanager.Secret(
                self,
                f"{key.title().replace('_', '')}Secret",
                secret_name=f"portolan/{self.config.name}/{suffix}",
                generate_secret_string=secretsmanager.SecretStringGenerator(
                    exclude_punctuation=True, password_length=40
                ),
            )
            for key, suffix in names.items()
        }

    def _auth_ecs_secrets(self) -> dict[str, ecs.Secret]:
        return {
            "PORTOLAN_ORG_KEY": ecs.Secret.from_secrets_manager(self.auth_secrets["org_key"]),
            "PORTOLAN_SESSION_SECRET": ecs.Secret.from_secrets_manager(
                self.auth_secrets["session_secret"]
            ),
            "PORTOLAN_MCP_ORG_KEY": ecs.Secret.from_secrets_manager(
                self.auth_secrets["mcp_org_key"]
            ),
        }

    # -- Edge TLS: ACM certificate on the ALB --------------------------------

    def _build_alb_certificate(self) -> acm.Certificate | None:
        """An ACM certificate for the ALB's HTTPS listener, so the
        CloudFront-to-ALB hop carries real TLS instead of plaintext
        (review finding F13: "the origin leg carries PHI unencrypted").

        `None` only when `config.domain_name` is unset AND `config.tier
        == "dev"` -- `_build_distribution` then leaves that one hop over
        plain HTTP, documented there. Every other tier raises: a real
        customer environment needs a real domain before it deploys, full
        stop, not a silently-plaintext origin leg.
        """
        if not self.config.domain_name:
            if self.config.tier == "dev":
                return None
            raise ValueError(
                f"customer {self.config.name!r} (tier {self.config.tier!r}): "
                "CustomerConfig.domain_name is required for every tier except "
                "'dev'. CloudFront cannot present real HTTPS to the ALB without "
                "an ACM certificate, and a certificate needs a real domain. Set "
                "domain_name in customers.yaml, or use tier 'dev' only for a "
                "domain-less local/demo synth."
            )
        return acm.Certificate(
            self,
            "AlbCertificate",
            domain_name=self.config.domain_name,
            validation=acm.CertificateValidation.from_dns(),
        )

    # -- ECS services -----------------------------------------------------

    def _build_web_service(
        self, sizing: TierSizing
    ) -> ecs_patterns.ApplicationLoadBalancedFargateService:
        image = ecs.ContainerImage.from_ecr_repository(self.repository, self.config.image_tag)
        tls_kwargs: dict = {}
        if self._alb_certificate is not None:
            tls_kwargs = {
                "certificate": self._alb_certificate,
                "protocol": elbv2.ApplicationProtocol.HTTPS,
                "redirect_http": True,
            }
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "WebService",
            cluster=self.cluster,
            cpu=sizing.web.cpu,
            memory_limit_mib=sizing.web.memory_mib,
            desired_count=1,
            # Public, per review finding F13: an internal ALB is
            # unreachable from CloudFront at all without a VPC origin
            # (this template used to synthesise clean and 502 on every
            # request). Restricting inbound traffic to CloudFront only
            # (rather than any public IP) is tracked in infra/README.md
            # "not done yet" -- it needs either a VPC origin (a newer
            # CloudFront/CDK feature) or a live AWS context lookup for
            # the managed CloudFront-origin-facing prefix list, and the
            # latter breaks `--dry` synth's placeholder-account-id
            # promise. The WAF web ACL in front of CloudFront is real
            # protection either way; this is the honestly-labelled gap,
            # not a silently-assumed one.
            public_load_balancer=True,
            **tls_kwargs,
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=image,
                container_name="web",
                container_port=8000,
                task_role=self.task_role,
                environment={
                    "PORTOLAN_TENANT": self.config.name,
                    "PORTOLAN_ROLE": "web",
                    "PORTOLAN_INGEST_QUEUE_URL": self.ingest_queue.queue_url,
                    "PORTOLAN_ORIGINALS_BUCKET": self.originals_bucket.bucket_name,
                    # Review finding F12: no LLM_PROVIDER here used to mean
                    # a deployed task fell back to Settings' local-dev
                    # default (agent_sdk, which shells out to a `claude`
                    # CLI login that doesn't exist in the container) rather
                    # than Bedrock.
                    "LLM_PROVIDER": "bedrock",
                },
                secrets={
                    "DATABASE_SECRET": ecs.Secret.from_secrets_manager(self.database.secret),
                    **self._auth_ecs_secrets(),
                },
                log_driver=ecs.LogDriver.aws_logs(
                    stream_prefix="web", log_group=self.log_group_web
                ),
            ),
        )
        self.database.connections.allow_default_port_from(service.service, "web service to Aurora")
        # Review finding F13: no ALB access logs at all -- the CloudFront
        # distribution below already logs every edge request, but nothing
        # logged what the ALB itself served or rejected.
        service.load_balancer.log_access_logs(self._access_logs_bucket, prefix="alb-access-logs/")
        return service

    def _build_worker_service(self, sizing: TierSizing) -> ecs.FargateService:
        image = ecs.ContainerImage.from_ecr_repository(self.repository, self.config.image_tag)
        task_definition = ecs.FargateTaskDefinition(
            self,
            "WorkerTaskDefinition",
            cpu=sizing.worker.cpu,
            memory_limit_mib=sizing.worker.memory_mib,
            task_role=self.task_role,
        )
        task_definition.add_container(
            "worker",
            image=image,
            environment={
                "PORTOLAN_TENANT": self.config.name,
                "PORTOLAN_ROLE": "worker",
                "PORTOLAN_INGEST_QUEUE_URL": self.ingest_queue.queue_url,
                "PORTOLAN_ORIGINALS_BUCKET": self.originals_bucket.bucket_name,
                "LLM_PROVIDER": "bedrock",  # review finding F12; see _build_web_service
            },
            secrets={
                "DATABASE_SECRET": ecs.Secret.from_secrets_manager(self.database.secret),
                **self._auth_ecs_secrets(),
            },
            logging=ecs.LogDriver.aws_logs(stream_prefix="worker", log_group=self.log_group_worker),
        )
        service = ecs.FargateService(
            self,
            "WorkerService",
            cluster=self.cluster,
            task_definition=task_definition,
            desired_count=sizing.worker_min_count,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
        )
        self.database.connections.allow_default_port_from(service, "worker service to Aurora")

        # Scaled by queue depth, per the build instructions: track the
        # ingest queue's visible-message count so a growing backlog adds
        # worker tasks and an empty queue scales back to worker_min_count.
        scaling = service.auto_scale_task_count(
            min_capacity=sizing.worker_min_count, max_capacity=sizing.worker_max_count
        )
        scaling.scale_on_metric(
            "ScaleOnQueueDepth",
            metric=self.ingest_queue.metric_approximate_number_of_messages_visible(
                period=Duration.minutes(1)
            ),
            scaling_steps=[
                appscaling.ScalingInterval(upper=0, change=-1),
                appscaling.ScalingInterval(lower=1, upper=50, change=0),
                appscaling.ScalingInterval(lower=50, change=1),
            ],
            cooldown=Duration.minutes(5),
        )
        return service

    def _grant_task_resources(self, role: iam.IRole) -> None:
        self.originals_bucket.grant_read_write(role)
        self.ingest_queue.grant_consume_messages(role)
        self.ingest_queue.grant_send_messages(role)
        self.database.secret.grant_read(role)
        self.kms_key.grant_encrypt_decrypt(role)

    # -- Edge: CloudFront + WAF --------------------------------------------

    def _build_web_acl(self) -> wafv2.CfnWebACL:
        # NOTE: a WAFv2 Web ACL with scope="CLOUDFRONT" must be created in
        # us-east-1 regardless of which region the distribution's origin
        # (this stack's ALB) lives in — a hard AWS constraint, not a CDK
        # limitation. This stack is deployed into `config.region`, which is
        # not always us-east-1, so as written this only produces a
        # deployable template when `config.region == "us-east-1"`. Splitting
        # the WAF+CloudFront pair into a companion stack pinned to us-east-1
        # (via `cross_region_references=True`, CDK 2.68+) is the correct
        # fix and is deliberately deferred — see infra/README.md "not done
        # yet". Flagged here rather than silently producing an
        # undeployable-outside-us-east-1 template with no explanation.
        return wafv2.CfnWebACL(
            self,
            "WebAcl",
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            scope="CLOUDFRONT",
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name=f"portolan-{self.config.name}-webacl",
                sampled_requests_enabled=True,
            ),
            name=f"portolan-{self.config.name}",
            rules=[
                wafv2.CfnWebACL.RuleProperty(
                    name="AWS-AWSManagedRulesCommonRuleSet",
                    priority=0,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS", name="AWSManagedRulesCommonRuleSet"
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name=f"portolan-{self.config.name}-common",
                        sampled_requests_enabled=True,
                    ),
                ),
                wafv2.CfnWebACL.RuleProperty(
                    name="AWS-AWSManagedRulesKnownBadInputsRuleSet",
                    priority=1,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS", name="AWSManagedRulesKnownBadInputsRuleSet"
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name=f"portolan-{self.config.name}-badinputs",
                        sampled_requests_enabled=True,
                    ),
                ),
                # Review finding F13: no rate-limit rule at all -- a
                # single IP blocked after 2,000 requests in a rolling
                # 5-minute window (WAF's own evaluation window for a
                # rate-based rule), which is generous for a real reader
                # and cheap insurance against a scraping or brute-force
                # burst against /login.
                wafv2.CfnWebACL.RuleProperty(
                    name="RateLimitPerIP",
                    priority=2,
                    action=wafv2.CfnWebACL.RuleActionProperty(block={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                            limit=2000, aggregate_key_type="IP"
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name=f"portolan-{self.config.name}-ratelimit",
                        sampled_requests_enabled=True,
                    ),
                ),
            ],
        )

    def _build_distribution(self) -> cloudfront.Distribution:
        # Review finding F13: this used to be hardcoded HTTP_ONLY,
        # meaning every request between CloudFront and the ALB -- PHI
        # included -- crossed the network in plaintext regardless of
        # whether the origin could have spoken TLS. HTTPS_ONLY whenever
        # `_build_alb_certificate` built a real certificate (every tier
        # except a domain-less "dev"); HTTP_ONLY is tolerated only in
        # that one case, which `_build_alb_certificate` itself restricts
        # to `tier == "dev"` -- this branch can't be reached by any other
        # tier without that method having already raised.
        origin_protocol = (
            cloudfront.OriginProtocolPolicy.HTTPS_ONLY
            if self._alb_certificate is not None
            else cloudfront.OriginProtocolPolicy.HTTP_ONLY
        )
        return cloudfront.Distribution(
            self,
            "Distribution",
            comment=f"Portolan {self.config.name}",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.LoadBalancerV2Origin(
                    self.web_service.load_balancer,
                    protocol_policy=origin_protocol,
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER,
            ),
            web_acl_id=self.web_acl.attr_arn,
            minimum_protocol_version=cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021,
            enable_logging=True,
            log_bucket=self._access_logs_bucket,
            log_file_prefix="cloudfront-access-logs/",
        )

    # -- Cognito -----------------------------------------------------------

    def _build_cognito(self) -> tuple[cognito.UserPool, cognito.UserPoolClient]:
        user_pool = cognito.UserPool(
            self,
            "UserPool",
            user_pool_name=f"portolan-{self.config.name}",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True, username=False),
            mfa=cognito.Mfa.OPTIONAL,
            mfa_second_factor=cognito.MfaSecondFactor(otp=True, sms=False),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )
        client = user_pool.add_client(
            "AppClient",
            user_pool_client_name=f"portolan-{self.config.name}-app",
            auth_flows=cognito.AuthFlow(user_srp=True),
            generate_secret=False,
        )
        return user_pool, client

    # -- Outputs -------------------------------------------------------

    def _outputs(self) -> None:
        CfnOutput(self, "DistributionDomainName", value=self.distribution.distribution_domain_name)
        CfnOutput(self, "OriginalsBucketName", value=self.originals_bucket.bucket_name)
        CfnOutput(self, "IngestQueueUrl", value=self.ingest_queue.queue_url)
        CfnOutput(self, "UserPoolId", value=self.user_pool.user_pool_id)
        CfnOutput(self, "DatabaseSecretArn", value=self.database.secret.secret_arn)  # type: ignore[union-attr]
