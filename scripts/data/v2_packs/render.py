"""YAML renderer for WP2's seven-plus-one vertical packs (`corpus/generator/
v2/packs/<vertical>.yaml`). Structural fields (products, roles, arcs,
topic_tree, fact_directives, persona_bank, decoy id/near_type/description,
name_pool_refs) come from `pack_data.py`; `claim_templates` (per fact
type, per event type) and each decoy's `example` text come from an
open-weight model via `claim_templates.py` and `scripts/data/
v2_gen_pack_templates.py` -- this module only lays the whole thing out as
YAML, byte for byte deterministic given the same input dict.

The arc/event skeleton below mirrors medical_devices.yaml's own EventType
-> FactType coverage shape exactly, so every pack satisfies the schema's
coverage validators (`_every_event_type_covered`, `_every_fact_type_
reachable`) by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

# repo_root/scripts/data/v2_packs/render.py -> repo_root is 3 parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "corpus" / "generator" / "v2" / "packs"

DIRECTIVES = [
    "hedged",
    "numbers_spelled_out",
    "colloquial",
    "embedded_mid_sentence",
    "passive",
    "through_an_alias",
    "corrected_mid_sentence",
    "list_item",
    "terse",
    "rambling",
    "formal",
    "via_quoted_reply",
    "second_language_phrasing",
]

FACT_TYPES = [
    "complaint",
    "request",
    "praise",
    "risk",
    "competitor_mention",
    "commitment",
    "decision",
    "contact_change",
    "pricing",
    "usage",
    "question",
    "other",
]

ARC_STEPS = {
    "rocky_go_live": [("go_live", 0), ("incident", 14), ("escalation", 30),
                       ("commitment_made", 35), ("commitment_due", 60), ("decision", 75)],
    "renewal_price_pushback": [("renewal_step", 0), ("price_negotiation", 10),
                                ("price_negotiation", 20), ("decision", 35)],
    "competitive_eval_save_or_churn": [("competitive_eval", 0), ("escalation", 15),
                                        ("commitment_made", 20), ("decision", 40)],
    "champion_transition": [("champion_departure", 0), ("qbr", 20), ("decision", 30)],
    "expansion": [("qbr", 0), ("expansion", 20), ("commitment_made", 25), ("decision", 40)],
    "incident_escalation_exec": [("incident", 0), ("incident", 7), ("escalation", 14),
                                  ("commitment_made", 18), ("commitment_due", 40)],
    "quiet_steady": [("qbr", 0), ("release", 30), ("qbr", 60), ("policy_change", 75)],
}  # fmt: skip

ARC_WEIGHTS = {
    "rocky_go_live": 1.2, "renewal_price_pushback": 1.3, "competitive_eval_save_or_churn": 1.0,
    "champion_transition": 1.0, "expansion": 1.0, "incident_escalation_exec": 1.1,
    "quiet_steady": 2.2,
}  # fmt: skip

EVENT_WEIGHTS = {
    "go_live": 1.0, "release": 1.0, "incident": 1.6, "escalation": 1.2, "qbr": 1.4,
    "renewal_step": 1.0, "price_negotiation": 0.9, "competitive_eval": 0.8,
    "champion_departure": 0.7, "expansion": 0.7, "commitment_made": 1.1,
    "commitment_due": 1.0, "decision": 0.9, "policy_change": 0.6,
}  # fmt: skip

# Shared claim-template skeleton (mirrors medical_devices.yaml's own event
# -> fact-type coverage exactly, guaranteeing every FactType is reachable
# and every EventType is covered). `<<key>>` tokens are this script's own
# substitution; `{account}`/`{product}`/etc. are the simulator's.
EVENT_CLAIMS_SKELETON: dict[str, dict[str, list[str]]] = {
    "go_live": {
        "usage": ["{account} brought the {product} live across its <<usage_unit>> as of {date}."],
        "complaint": [
            "{account}'s go-live on {product} slipped its planned date and <<staff_word>> "
            "reported <<setup_friction>> as of {date}."
        ],
    },
    "release": {
        "usage": [
            "{vendor} shipped a {product} <<release_word>> that {account} installed as of {date}."
        ],
        "praise": [
            "{account}'s <<ops_role>> called the {product} update \"the smoothest rollout "
            "we've had\" as of {date}."
        ],
    },
    "incident": {
        "complaint": [
            "{account} reported a {product} <<incident_word>> affecting <<impact_area>> as of {date}."
        ],
        "risk": [
            "{account}'s <<clinical_role>> raised <<risk_topic>> concerns about {product} after the {date} incident."
        ],
    },
    "escalation": {
        "risk": [
            "{account}'s CFO escalated the open {product} incident to {vendor} leadership as of {date}."
        ],
        "complaint": [
            "{account} said the {product} incident response has been too slow as of {date}."
        ],
    },
    "qbr": {
        "usage": [
            "{account} runs {product} across its <<usage_scope>> as reported at the {date} QBR."
        ],
        "praise": [
            "{account}'s <<champion_role>> praised {product} <<praise_metric>> at the {date} QBR."
        ],
        "question": [
            "{account} asked whether {product} supports a planned <<integration_target>>; unanswered as of {date}."
        ],
    },
    "renewal_step": {
        "decision": ["{account} chose to renew its {vendor} contract for one year as of {date}."],
        "pricing": ["{account} expects a volume discount on its {date} {vendor} renewal."],
    },
    "price_negotiation": {
        "pricing": [
            "{account}'s CFO pushed back on the proposed {product} renewal price as of {date}."
        ],
        "request": ["{account} asked {vendor} for a multi-year price lock as of {date}."],
    },
    "competitive_eval": {
        "competitor_mention": [
            "{account} is evaluating a {competitor} <<product_noun>> alongside {product} as of {date}."
        ],
        "risk": [
            "{account}'s <<budget_role>> said budget could shift to {competitor} if service does not improve, as of {date}."
        ],
    },
    "champion_departure": {
        "contact_change": [
            "{person} left {account} and {vendor} has not identified a replacement champion as of {date}."
        ],
        "risk": ["{account}'s relationship is at risk after {person}'s departure as of {date}."],
    },
    "expansion": {
        "decision": [
            "{account} chose to expand its {product} deployment to a <<expansion_scope>> as of {date}."
        ],
        "usage": ["{account} added {product} to its <<expansion_area>> as of {date}."],
    },
    "commitment_made": {
        "commitment": [
            "{vendor}'s rep promised {account} a {product} <<remediation_word>> by {date}."
        ],
    },
    "commitment_due": {
        "commitment": ["{vendor} owes {account} the {product} fix committed for {date}."],
        "risk": [
            "{account} flagged that {vendor}'s {date} commitment on {product} is now overdue."
        ],
    },
    "decision": {
        "decision": [
            "{account} decided to standardize on {product} across its <<facilities_word>> as of {date}."
        ],
    },
    "policy_change": {
        "other": [
            "{account} updated its <<policy_area>> policy for {product} purchases as of {date}."
        ],
        "decision": [
            "{account}'s board approved the updated {product} <<policy_area>> policy as of {date}."
        ],
    },
}

# event_type -> [fact_type, ...] in the same order as EVENT_CLAIMS_SKELETON
# above -- the shape every pack's event_templates renders in, independent
# of which stage supplied the claim-template TEXT for each (vocab-
# substitution originally, a model-generated `claim_templates` dict now).
EVENT_TYPE_FACT_TYPES: dict[str, list[str]] = {
    event_type: list(per_type.keys()) for event_type, per_type in EVENT_CLAIMS_SKELETON.items()
}


def _sub(template: str, vocab: dict[str, str]) -> str:
    out = template
    for key, value in vocab.items():
        out = out.replace(f"<<{key}>>", value)
    if "<<" in out:
        raise ValueError(f"unsubstituted vocab token in: {out!r}")
    return out


def _yq(s: str) -> str:
    """Double-quote a YAML scalar, JSON-escaped (JSON string syntax is a
    valid subset of YAML double-quoted scalar syntax)."""
    return json.dumps(s)


def _flow_list(items: list[str]) -> str:
    return json.dumps(items)


def render_pack(v: dict) -> str:
    lines: list[str] = []
    lines.append(f"# {v['name']} vertical pack -- WP2 (`01-spec/world-generator-v2-2026-09-22.md`")
    lines.append(
        "# section 4). Validates against `corpus/generator/v2/packs/schema.py::VerticalPack`."
    )
    if v.get("held_out"):
        lines.append("#")
        lines.append("# Held-out vertical (spec section 4's own example: \"an eighth held-out")
        lines.append("# vertical (for example lab and imaging services) to measure how well the")
        lines.append('# student generalizes past its training verticals"). Drawn only for')
        lines.append("# test-split worlds -- never a train or validation world.")
    if v.get("guard_note"):
        lines.append("#")
        lines.append(f"# {v['guard_note']}")
    lines.append("")
    lines.append(f"id: {v['id']}")
    lines.append(f"name: {v['name']}")
    lines.append(f"vendor_what: {_yq(v['vendor_what'])}")
    lines.append("")

    lines.append("products:")
    for p in v["products"]:
        lines.append(f"  - kind: {p['kind']}")
        lines.append(f"    name_template: {_yq(p['name_template'])}")
        lines.append(f"    tiers: {_flow_list(p.get('tiers', []))}")
        issues = p.get("known_issue_pool", [])
        if issues:
            lines.append("    known_issue_pool:")
            for iss in issues:
                lines.append(f"      - {_yq(iss)}")
        else:
            lines.append("    known_issue_pool: []")
    lines.append("")

    lines.append("roles:")
    for r in v["roles"]:
        lines.append(f"  - title: {_yq(r['title'])}")
        lines.append(f"    seniority: {r['seniority']}")
        lines.append(f"    is_champion_candidate: {'true' if r.get('champion') else 'false'}")
    lines.append("")

    # `claim_templates`: {fact_type: {event_type: [sentence, ...]}}, model-
    # generated (coordinator review of 7f779a3: `Fact.claim` labels must
    # come from an open-weight teacher, never be Claude-written -- see
    # `gen_claims.py`/`run_all.py`). `EVENT_TYPE_FACT_TYPES` below is just
    # the event_type -> fact_type *shape* (still mirrors medical_devices.
    # yaml's own wiring), not text.
    lines.append("event_templates:")
    claim_templates = v["claim_templates"]
    for event_type, fact_types in EVENT_TYPE_FACT_TYPES.items():
        lines.append(f"  - event_type: {event_type}")
        lines.append(f"    weight: {EVENT_WEIGHTS[event_type]}")
        lines.append("    claim_templates:")
        for fact_type in fact_types:
            templates = claim_templates[fact_type][event_type]
            if len(templates) == 1:
                lines.append(f"      {fact_type}: [{_yq(templates[0])}]")
            else:
                lines.append(f"      {fact_type}:")
                for t in templates:
                    lines.append(f"        - {_yq(t)}")
    lines.append("")

    lines.append("arcs:")
    for arc_type, steps in ARC_STEPS.items():
        lines.append(f"  - id: {arc_type}_v1")
        lines.append(f"    arc_type: {arc_type}")
        lines.append(f"    weight: {ARC_WEIGHTS[arc_type]}")
        lines.append("    steps:")
        for event_type, offset in steps:
            lines.append(f"      - {{event_type: {event_type}, day_offset: {offset}}}")
    lines.append("")

    lines.append("topic_tree:")
    for node in v["topic_tree"]:
        lines.append(f"  - name: {_yq(node['name'])}")
        if node.get("children"):
            lines.append("    children:")
            for c in node["children"]:
                lines.append(f"      - name: {_yq(c)}")
    lines.append("")

    lines.append("# One shared generic directive list, aliased into every fact type (section 4,")
    lines.append('# "Phrasing": a phrasing-style vocabulary applied across types, not')
    lines.append("# type-specific content -- same pattern as medical_devices.yaml).")
    lines.append("fact_directives:")
    first = FACT_TYPES[0]
    lines.append(f"  {first}: &directives")
    for d in DIRECTIVES:
        lines.append(f"    - {d}")
    for ft in FACT_TYPES[1:]:
        lines.append(f"  {ft}: *directives")
    lines.append("")

    lines.append("persona_bank:")
    for p in v["personas"]:
        lines.append(f"  - id: {p['id']}")
        lines.append(f"    speech_register: {_yq(p['register'])}")
        lines.append(f"    formality: {p['formality']}")
        lo, hi = p["typo"]
        lines.append(f"    typo_rate_range: [{lo}, {hi}]")
        lines.append(f"    chat_style: {p['chat_style']}")
        lines.append(f"    signature_style: {p['signature']}")
        lines.append(f"    hedging: {'true' if p['hedging'] else 'false'}")
        if p.get("second_language_features"):
            lines.append(
                f"    second_language_features: {_flow_list(p['second_language_features'])}"
            )
    lines.append("")

    lines.append("decoy_classes:")
    for d in v["decoys"]:
        lines.append(f"  - id: {d['id']}")
        lines.append(f"    near_type: {d['near_type']}")
        lines.append(f"    description: {_yq(d['description'])}")
        lines.append(f"    example: {_yq(d['example'])}")
    lines.append("")

    npr = v["name_pool_refs"]
    lines.append("name_pool_refs:")
    lines.append(f"  account_suffixes_system: {_flow_list(npr['system'])}")
    lines.append(f"  account_suffixes_facility: {_flow_list(npr['facility'])}")
    lines.append(f"  account_suffixes_standalone: {_flow_list(npr['standalone'])}")
    lines.append('  vendor_domain_tld: ".com"')
    lines.append("")

    excluded = v.get("excluded_product_kinds", [])
    lines.append(f"excluded_product_kinds: {_flow_list(excluded)}")
    lines.append(f"held_out: {'true' if v.get('held_out') else 'false'}")
    lines.append("")

    src = v["claim_templates_source"]
    lines.append("# Coordinator review of 7f779a3, 2026-09-22: every claim template above and")
    lines.append("# every decoy_classes[].example was written by this open-weight model, curated")
    lines.append("# (filtered/deduped, dropped when invalid, never rephrased) by scripts/data/")
    lines.append("# v2_gen_pack_templates.py -- not Claude-written, since this text reaches")
    lines.append("# training data. raw_response_sha256 points at the exact, unedited model")
    lines.append("# responses under var/data/v2_packs/raw/ (gitignored, not committed).")
    lines.append("claim_templates_source:")
    lines.append(f"  model: {_yq(src['model'])}")
    lines.append(f"  generated: {_yq(src['generated'])}")
    lines.append("  raw_response_sha256:")
    for digest in src["raw_response_sha256"]:
        lines.append(f"    - {_yq(digest)}")
    lines.append("")

    return "\n".join(lines)
