"""Per-tier sizing table for Portolan customer infrastructure.

Source: `01-spec/cost-analysis-2026-09-12.md` section 2 ("Assumptions",
rows 18 and 20) and section 3 ("Per-customer monthly cost"). Every number
below traces to those two sections; re-derive from a revised cost analysis
rather than eyeballing a change here.

Aurora Serverless v2 capacity — cost analysis section 2, assumption 18
(marked **assumed** in that document itself):

    Small:  0.5 ACU idle, bursts to 2 ACU during a ~4h nightly ingest window (avg 0.75)
    Medium: 1 ACU idle, bursts to 4 ACU over ~6h                              (avg 1.75)
    Large:  2 ACU idle, bursts to 8 ACU over ~8h                             (avg 4.0)

Every tier's idle-to-burst ratio is 4x. That ratio is reused below to size
the ingest worker's autoscaling range, for lack of any other burst signal
in the spec (see the Fargate note below).

Fargate compute — cost analysis section 2, assumption 20: "Fargate on
Graviton, always on: Small 1 vCPU / 2 GB total; Medium 3 / 6; Large 10 / 20.
Plus one ALB." That is a **combined web+worker total per tier**, not a
per-service split — the cost analysis doesn't distinguish the two ECS
services that `01-spec/build-contracts.md` ("Infra") calls for (`web`
behind an internal ALB, `worker` scaled by queue depth). The split into
`web` / `worker` sizes below is this module's engineering judgment, not a
number sourced from the cost analysis:

- `web` is sized as a small, fixed always-on task (chat/API traffic is
  steady, not bursty, per spec section 10).
- `worker` is sized to fit the remaining budget at `worker_min_count`, and
  autoscales on SQS queue depth (build-contracts.md) up to
  `worker_max_count`, using the same ~4x idle-to-burst ratio Aurora's ACUs
  use above, since both are sized around the same nightly ingest window.

Revisit this split once a real deployment measures actual worker
utilization — nothing in the spec or the cost analysis pins it down further.

All CPU/memory pairs below are valid Fargate task-size combinations
(cpu units where 1024 = 1 vCPU; memory in MiB).
"""

from __future__ import annotations

from dataclasses import dataclass

COST_ANALYSIS_SOURCE = "01-spec/cost-analysis-2026-09-12.md §2 (assumptions 18, 20), §3"


@dataclass(frozen=True)
class FargateServiceSize:
    """One ECS Fargate task's CPU/memory shape.

    `cpu` is in Fargate CPU units (1024 = 1 vCPU); `memory_mib` in MiB. Must
    be one of the CPU/memory pairs Fargate accepts.
    """

    cpu: int
    memory_mib: int


@dataclass(frozen=True)
class TierSizing:
    """The full sizing table entry for one customer tier."""

    tier: str
    aurora_min_acu: float
    aurora_max_acu: float
    web: FargateServiceSize
    worker: FargateServiceSize
    worker_min_count: int
    worker_max_count: int
    source: str = COST_ANALYSIS_SOURCE


TIERS: dict[str, TierSizing] = {
    # Fix wave 2026-09-12 (review finding F13). Same shape as "small" --
    # this tier exists to make a domain-less synth (local dev, a demo
    # tenant with no real customer domain yet) an explicit, deliberate
    # choice rather than an accident. `PortolanCustomerStack` only lets
    # `CustomerConfig.domain_name` be unset for this one tier; every
    # other tier fails synth outright without a real domain, because
    # CloudFront cannot present real HTTPS to the ALB without an ACM
    # certificate, and a certificate needs a real domain.
    "dev": TierSizing(
        tier="dev",
        aurora_min_acu=0.5,
        aurora_max_acu=2.0,
        web=FargateServiceSize(cpu=256, memory_mib=512),
        worker=FargateServiceSize(cpu=256, memory_mib=512),
        worker_min_count=1,
        worker_max_count=2,
        source="fix wave 2026-09-12, review finding F13 (not from the cost analysis: "
        "same shape as 'small', for a domain-less local/demo synth only)",
    ),
    "small": TierSizing(
        tier="small",
        aurora_min_acu=0.5,
        aurora_max_acu=2.0,
        web=FargateServiceSize(cpu=256, memory_mib=512),
        worker=FargateServiceSize(cpu=256, memory_mib=512),
        worker_min_count=1,
        worker_max_count=2,
    ),
    "medium": TierSizing(
        tier="medium",
        aurora_min_acu=1.0,
        aurora_max_acu=4.0,
        web=FargateServiceSize(cpu=512, memory_mib=1024),
        worker=FargateServiceSize(cpu=512, memory_mib=1024),
        worker_min_count=1,
        worker_max_count=4,
    ),
    "large": TierSizing(
        tier="large",
        aurora_min_acu=2.0,
        aurora_max_acu=8.0,
        web=FargateServiceSize(cpu=1024, memory_mib=2048),
        worker=FargateServiceSize(cpu=2048, memory_mib=4096),
        worker_min_count=1,
        worker_max_count=8,
    ),
}

VALID_TIERS = tuple(TIERS.keys())


def get_tier(name: str) -> TierSizing:
    """Look up a tier by name (`"small" | "medium" | "large"`).

    Raises `ValueError` naming the valid options, rather than a bare
    `KeyError`, since this is read from `customers.yaml` and a typo there
    should fail loud with a fixable message.
    """
    try:
        return TIERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown tier {name!r}; expected one of {VALID_TIERS}") from exc
