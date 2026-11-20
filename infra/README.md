# Infra

AWS CDK (Python), account-per-customer. See `01-spec/portolan-spec-v2.md`
§11 ("Deployment and operations") and §12 ("Security and compliance") for
the design this implements, `01-spec/adr-001-language-and-stack.md` for why
CDK-in-Python rather than Terraform or raw CloudFormation, and
`01-spec/cost-analysis-2026-09-12.md` §2/§3 for where the tier sizing in
`portolan_infra/tiers.py` comes from.

**Not deployed.** There is no AWS account, let alone an AWS Organization,
yet. Everything here synthesises (`App().synth()`,
`aws_cdk.assertions.Template`) and is unit-tested; nothing has been run
against a real account.

## What is here

```
infra/
  app.py                          entry point: reads customers.yaml, builds one App
  cdk.json                        `cdk` CLI config (app command), for once Node/the CDK CLI are set up
  customers.yaml                  the customer list (see "How to add a customer")
  portolan_infra/
    tiers.py                      per-tier sizing table (ACU, Fargate, worker autoscaling), sourced from the cost analysis
    bedrock_models.py             the Bedrock model ids the ingest pipeline's IAM role is scoped to
    customer_stack.py             PortolanCustomerStack: one full, isolated customer environment
    platform_stack.py             PortolanPlatformStack: the shared org-level security/CI account
  deploy/
    snapshot.py                   create + wait for an Aurora cluster snapshot
    smoke.py                      post-deploy control query + note/source count check
    canary.py                     deploy order: demo, canary, the rest
    rollback.py                   redeploy the previous image tag
```

Every script under `deploy/` takes `--dry-run` and cites spec §11 in its
docstring. None of them touch AWS, a database, or the network in
`--dry-run` mode.

### `PortolanCustomerStack` (per customer, single-tenant)

VPC (2 AZs, one NAT gateway, public/app/data subnet tiers) · Aurora
PostgreSQL Serverless v2 (engine 17, sized by tier, encrypted with a
customer-specific KMS key, deletion protection on, 35-day backups, no
public access, credentials in Secrets Manager) · S3 bucket for originals
(versioned, SSE-KMS, public access blocked, optional Object Lock governance
mode, Intelligent-Tiering after 90 days, access logging to a separate
SSE-S3 bucket) · SQS ingest queue + a 14-day-retention dead-letter queue ·
ECS Fargate cluster running `web` and `worker` (autoscaled on the ingest
queue's depth) from the shared ECR image · CloudFront in front of the
ALB, over HTTPS end to end (an ACM certificate on the ALB, real for any
tier with `CustomerConfig.domain_name` set; the "dev" tier is the one
exception allowed to stay HTTP-only when no domain is configured yet --
see `customer_stack.py`'s `_build_alb_certificate`) with AWS-managed WAF
rules plus a per-IP rate limit · Cognito user pool (email login, MFA
optional) · 1-year-retention CloudWatch log groups · an IAM task role
(granted its resources exactly once, not once per service) scoped to
Bedrock `InvokeModel` on the model ids in
`01-spec/ingest-cost-and-models.md`, the originals bucket, the ingest
queue, and the database secret, plus three stack-generated Secrets
Manager secrets (`PORTOLAN_ORG_KEY`, `PORTOLAN_SESSION_SECRET`,
`PORTOLAN_MCP_ORG_KEY` -- review finding F1) injected into both services'
task definitions. Every resource CDK can tag carries `portolan:customer`
/ `portolan:tier`.

### `PortolanPlatformStack` (the shared account)

ECR repository (image scanning on push) · a GitHub OIDC provider and a
deploy role trusted only by `maintainertetik-creator/portolan-oss` on `main` ·
a CloudTrail organization trail to a locked (versioned + Object Lock)
bucket · a GuardDuty detector · an AWS Config recorder with the
`Operational-Best-Practices-for-HIPAA-Security` conformance pack ·
Security Hub with the **AWS Foundational** standard.

Why AWS Foundational and not "the HIPAA standard": Security Hub does not
have one. Its built-in standards are AWS Foundational, AI Security Best
Practices, Resource Tagging, CIS, NIST 800-53, NIST 800-171, and PCI DSS —
see `01-spec/research/research-aws-baa-2026-09-12.md` §4. HIPAA-mapped
checks come from the Config conformance pack instead.

## Bootstrap steps for a new AWS Organization

1. Form the legal entity and get the EIN before any customer paperwork —
   AWS itself doesn't require this to open an account, but no serious
   healthcare buyer contracts with an individual (research doc §7).
2. Create the AWS Organization's management account under the company's
   legal name; turn on "all features" in AWS Organizations.
3. **Accept the Business Associate Addendum in AWS Artifact, at the
   organization level, from the management account.** Free, takes minutes,
   and covers every member account automatically — including ones added
   later. Do this before any customer account exists (research doc §1, §8).
4. Deploy `PortolanPlatformStack` into the management account (or a
   delegated security-tooling account): CloudTrail org trail, GuardDuty,
   the Config HIPAA conformance pack, Security Hub, the ECR repo, the
   GitHub OIDC deploy role.
5. Create the first customer's member account under the Organization;
   confirm AWS flags it a HIPAA account (it should be, automatically, per
   step 3).
6. Set `PORTOLAN_PLATFORM_ACCOUNT_ID` and
   `PORTOLAN_ACCOUNT_<CUSTOMER_NAME>` (see below) and deploy that
   customer's `PortolanCustomerStack` into their account.
7. Get cyber liability insurance quotes and start SOC 2 readiness in
   parallel — buyers ask before contract signature (research doc §6, §9).

## How to add a customer

One entry in `infra/customers.yaml`:

```yaml
- name: acme
  tier: medium       # dev | small | medium | large -- see portolan_infra/tiers.py
  region: us-east-1
  domain_name: acme.customers.portolan.example   # required for every tier except "dev"
```

`domain_name` is required for every tier except `"dev"` (review finding
F13): CloudFront presents real HTTPS to the ALB via an ACM certificate,
and a certificate needs a real domain. `"dev"` is the one tier allowed to
skip it, for a domain-less local/demo synth -- `PortolanCustomerStack`
fails synth immediately, with a message naming the customer, if a
non-`"dev"` tier has no `domain_name`.

Then, before a real (`--no-dry`) deploy, set that customer's AWS account id
so `infra/app.py` can find it:

```bash
export PORTOLAN_ACCOUNT_ACME=123456789012   # name upper-cased, non-alnum -> "_"
```

`infra/app.py --dry` (the default) never needs this — it fills in a
placeholder account id so the stack can still synthesise and be tested.

## How a release flows

1. CI builds the image from `docker/Dockerfile` and pushes it to the shared
   ECR repo (`PortolanPlatformStack.repository`), tagged with the commit SHA.
2. `infra/deploy/canary.py --image-tag <sha>` runs the rollout: for each
   customer, in order —
   1. `snapshot.py` snapshots that customer's Aurora cluster and waits for
      it to be ready.
   2. `smoke.py baseline` records the pre-deploy control query result and
      note/source counts.
   3. The CDK stack deploys with `PORTOLAN_IMAGE_TAG=<sha>`.
   4. `smoke.py check` re-runs the control query and counts; a failure (a
      dropped count, or a broken control query) stops the rollout.
3. Order: the `demo` tenant first, then one canary customer, then
   everyone else (spec §11). `canary.py` reads this order straight out of
   `customers.yaml`.
4. On a failed smoke check, `rollback.py --customer <name> --bad-tag <sha>`
   finds the tag that was deployed immediately before `<sha>` in ECR's push
   history and redeploys that customer pinned to it.

Migrations are additive-only across this whole flow (expand, migrate,
contract as separate releases) — enforced by review, not by these scripts.

## What is deliberately not done yet

- **No `cdk deploy` has ever run.** There is no AWS account. `deploy/*.py`
  shell out to the `cdk` CLI assuming credentials and the CLI are already
  set up; none of that is wired to real credentials.
- **The ALB is public, reachable directly, not just through CloudFront.**
  Fix wave 2026-09-12 (review finding F13) made it public so CloudFront
  could reach it at all (an internal ALB synthesised clean and served
  502s on every request -- CloudFront cannot reach a private-IP origin
  without a VPC origin) and put a real ACM certificate + HTTPS on it, so
  the origin leg no longer carries PHI in plaintext. Restricting inbound
  traffic to CloudFront only needs either a VPC origin (a newer
  CloudFront/CDK feature) or a live AWS context lookup for the managed
  CloudFront-origin-facing prefix list -- the latter breaks `--dry`
  synth's placeholder-account-id promise, so it's not wired in. The WAF
  web ACL in front of CloudFront (managed rules + a per-IP rate limit)
  is real protection either way; a direct hit on the ALB bypasses it.
- **CloudFront + WAF cross-region constraint, unresolved.** A WAFv2 Web ACL
  with `scope="CLOUDFRONT"` must be created in `us-east-1`, regardless of
  which region the distribution's origin (the customer's ALB) lives in —
  a hard AWS constraint. `customer_stack.py` currently creates the Web ACL
  in the same stack/region as everything else, which only produces a
  deployable template when that customer's region is `us-east-1`. The
  correct fix is a companion stack pinned to `us-east-1` with
  `cross_region_references=True` (CDK 2.68+); deferred until a customer in
  a non-`us-east-1` region is real. Flagged in `customer_stack.py` at the
  Web ACL construction site too.
- **The GitHub deploy role has no deploy permissions attached.** It trusts
  `maintainertetik-creator/portolan-oss`@`main` via OIDC, but scoping an actual
  `cdk deploy`-shaped policy makes sense once there is a second AWS account
  to bound it to — attaching broad access now, before any customer account
  exists, is the wrong default for a HIPAA-adjacent org.
- **Some Bedrock model ids are unverified this session.** `bedrock_models.py`
  documents which ones: Nova Lite and GPT-5.6 Luna are independently
  confirmed against a live AWS model-card page (in
  `portolan/extract/backends.py`); Nova Micro, Nova Pro, gpt-oss-20b/120b,
  and Titan Text Embeddings V2 follow AWS's naming convention but were not
  re-fetched live. Re-confirm all of them before scoping a real customer's
  IAM policy.
- **RBAC:** out of scope for v1 everywhere in this repo (build-contracts.md
  "Ground rules"), infra included — Cognito here is authentication only.
- **Macie, Audit Manager, SOC 2 tooling:** not part of this stack. Macie is
  a candidate addition once real customer data exists to scan; Audit
  Manager is unusable for a new account as of 2026-04-30 (research doc §4);
  SOC 2 is a business process, not infrastructure.
- **No cost or budget alarms.** The cost analysis gives per-tier estimates;
  nothing here enforces them yet (no AWS Budgets, no anomaly detection).
