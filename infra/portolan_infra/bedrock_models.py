"""Bedrock model ids the ingest pipeline's IAM task role is scoped to invoke.

Model *names and prices* come from `01-spec/ingest-cost-and-models.md` (the
extraction cost table and the "Bedrock options" sweep) and, for Titan,
`01-spec/cost-analysis-2026-09-12.md` §2 assumption 9. Qwen3.8-27B is
deliberately excluded: per `ingest-cost-and-models.md`, it is "not on
Bedrock's managed list" (Custom Model Import only, unverified).

Two ids are independently verified against a live AWS model-card page
already, in `portolan/extract/backends.py` (Nova Lite, GPT-5.6 Luna) — they
are imported from there rather than retyped so there is exactly one source
of truth for them. The Claude family ids match what
`portolan.extract.backends.AnthropicBedrockBackend` and its tests already
use (`tests/test_backends.py::test_infer_model_key_...`, and the
`AnthropicBedrockBackend("anthropic.claude-haiku-4-5", ...)` construction in
that same file): the bare `anthropic.claude-*` id, no cross-region
inference-profile prefix.

The remaining ids (Nova Micro, Nova Pro, gpt-oss-20b, gpt-oss-120b, Titan
Text Embeddings V2) follow AWS's documented Bedrock model-id naming
convention but were **not** independently re-fetched from a live model-card
page in this session — per the project's "verify before claiming" rule,
re-confirm each at
https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html before
this list is used to scope a real IAM policy for a paying customer.
"""

from __future__ import annotations

from portolan.extract.backends import GPT_5_6_LUNA_MODEL_ID, NOVA_LITE_MODEL_ID

# Verified against a live AWS model-card page — see backends.py's own
# citations for the fetch. Re-exported here under names that read clearly
# next to the unverified ones below.
NOVA_LITE = NOVA_LITE_MODEL_ID
GPT_5_6_LUNA = GPT_5_6_LUNA_MODEL_ID

# Already load-bearing elsewhere in this repo (backends.py's model-key
# inference, pricing.py's PRICING table, test_backends.py's construction
# test) — not re-fetched here, but not a fresh guess either.
CLAUDE_HAIKU_4_5 = "anthropic.claude-haiku-4-5"
CLAUDE_SONNET_5 = "anthropic.claude-sonnet-5"
CLAUDE_OPUS_5 = "anthropic.claude-opus-5"

# Naming-convention guesses, NOT independently re-fetched this session.
NOVA_MICRO = "amazon.nova-micro-v1:0"
NOVA_PRO = "amazon.nova-pro-v1:0"
GPT_OSS_20B = "openai.gpt-oss-20b-1:0"
GPT_OSS_120B = "openai.gpt-oss-120b-1:0"
TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"

# Every model the ingest pipeline (extraction + merge compare + embedding)
# is allowed to call, per ingest-cost-and-models.md's extraction table and
# Bedrock options sweep, plus Titan for embeddings.
EXTRACTION_AND_COMPARE_MODEL_IDS: list[str] = [
    NOVA_MICRO,
    NOVA_LITE,
    GPT_OSS_20B,
    GPT_OSS_120B,
    GPT_5_6_LUNA,
    NOVA_PRO,
    CLAUDE_HAIKU_4_5,
    CLAUDE_SONNET_5,
    CLAUDE_OPUS_5,
    TITAN_EMBED_V2,
]

# Model ids that route through a cross-region inference profile rather than
# a plain on-demand foundation-model id (see backends.py's citation for why
# GPT-5.6 Luna needs this). Such ids resolve to an `inference-profile` ARN
# in IAM, not `foundation-model`.
_INFERENCE_PROFILE_PREFIXES = ("us.", "eu.", "apac.", "global.")


def bedrock_model_arns(region: str, account: str) -> list[str]:
    """Foundation-model (and, where needed, inference-profile) ARNs for
    every id in `EXTRACTION_AND_COMPARE_MODEL_IDS`, in `region`/`account`.

    A plain on-demand id (e.g. `amazon.nova-lite-v1:0`) is an account-less,
    region-scoped `foundation-model` resource. A cross-region inference
    profile id (e.g. `us.openai.gpt-5.6-luna`) is an account-scoped
    `inference-profile` resource instead. Both ARN shapes are granted for
    every id so `bedrock:InvokeModel` works whichever routing that id
    needs, without the caller having to know which is which.
    """
    arns: list[str] = []
    for model_id in EXTRACTION_AND_COMPARE_MODEL_IDS:
        arns.append(f"arn:aws:bedrock:{region}::foundation-model/{model_id}")
        if model_id.startswith(_INFERENCE_PROFILE_PREFIXES):
            arns.append(f"arn:aws:bedrock:{region}:{account}:inference-profile/{model_id}")
    return arns
