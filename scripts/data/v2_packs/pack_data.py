"""Per-vertical STRUCTURAL content for WP2's seven packs (products, buyer/
vendor roles, topic tree, persona bank, decoy id/near_type/description,
name-pool suffixes) -- hand-authored, per the coordinator's 2026-09-22
review of 7f779a3: "Structural fields (roles, products, arcs, directives,
personas, topic-tree labels) may stay as you wrote them." Only the
`claim_templates` text and each decoy's `example` text are model-
generated (`claim_templates.py` / `scripts/data/v2_gen_pack_templates.py`)
-- this module supplies everything else, fed through `render.render_pack`
for the mechanical YAML-shape work (schema coverage, quoting, arcs
skeleton). `medical_devices.yaml` (WP1's own pack) is edited in place by
`scripts/data/v2_gen_pack_templates.py` instead of going through this
module, so its structural content -- written by WP1, not WP2 -- is never
touched."""

from __future__ import annotations

COMMON_DECOY_SHAPE = [
    # (id, near_type) pairs mirrored from medical_devices.yaml's seven
    # classes, plus an eighth: an explicit commitment/decision confusable
    # pair (WP2 task: "type-confusable pairs such as commitment versus
    # decision").
    ("hypothetical", "risk"),
    ("negation", "competitor_mention"),
    ("hearsay_other_company", "competitor_mention"),
    ("generic_industry_talk", "other"),
    ("rhetorical_question", "question"),
    ("vendor_internal_speculation", "risk"),
    ("resolved_issue_as_history", "complaint"),
    ("commitment_decision_confusable", "decision"),
]


def personas(register: str, id_prefix: str) -> list[dict]:
    """Ten personas, register-flavored per vertical -- same diversity
    shape as medical_devices.yaml's own bank (terse/rambling/formal-exec/
    typo-prone/emoji/second-language/formal-specialist/casual-IT/hedging/
    direct-CFO), just relabeled per vertical's primary clinical/ops
    register so no two packs' persona ids collide."""
    return [
        {
            "id": f"terse_{id_prefix}",
            "register": register,
            "formality": "neutral",
            "typo": (0.0, 0.01),
            "chat_style": "short_fragments",
            "signature": "initials_only",
            "hedging": False,
        },
        {
            "id": f"rambling_ops_{id_prefix}",
            "register": "operations",
            "formality": "informal",
            "typo": (0.01, 0.03),
            "chat_style": "long_threads",
            "signature": "full_name_title",
            "hedging": True,
        },
        {
            "id": f"formal_executive_{id_prefix}",
            "register": "executive",
            "formality": "formal",
            "typo": (0.0, 0.0),
            "chat_style": "rare_and_brief",
            "signature": "full_block",
            "hedging": False,
        },
        {
            "id": f"typo_prone_frontline_{id_prefix}",
            "register": register,
            "formality": "informal",
            "typo": (0.03, 0.06),
            "chat_style": "emoji_light",
            "signature": "first_name_only",
            "hedging": False,
        },
        {
            "id": f"emoji_chat_{id_prefix}",
            "register": register,
            "formality": "informal",
            "typo": (0.01, 0.02),
            "chat_style": "emoji_heavy",
            "signature": "first_name_only",
            "hedging": False,
        },
        {
            "id": f"second_language_procurement_{id_prefix}",
            "register": "procurement",
            "formality": "formal",
            "typo": (0.01, 0.02),
            "chat_style": "measured",
            "signature": "full_block",
            "hedging": True,
            "second_language_features": ["article_drop", "present_tense_for_future"],
        },
        {
            "id": f"formal_specialist_{id_prefix}",
            "register": f"{register}_specialist",
            "formality": "formal",
            "typo": (0.0, 0.01),
            "chat_style": "precise",
            "signature": "full_name_title",
            "hedging": False,
        },
        {
            "id": f"casual_it_{id_prefix}",
            "register": "operations",
            "formality": "informal",
            "typo": (0.02, 0.04),
            "chat_style": "short_fragments",
            "signature": "none",
            "hedging": False,
        },
        {
            "id": f"hedging_manager_{id_prefix}",
            "register": "operations",
            "formality": "neutral",
            "typo": (0.0, 0.02),
            "chat_style": "measured",
            "signature": "first_name_only",
            "hedging": True,
        },
        {
            "id": f"direct_cfo_{id_prefix}",
            "register": "executive",
            "formality": "formal",
            "typo": (0.0, 0.0),
            "chat_style": "rare_and_brief",
            "signature": "full_block",
            "hedging": False,
        },
    ]


def decoys(id_prefix: str, examples: dict[str, str]) -> list[dict]:
    descriptions = {
        "hypothetical": "A hypothetical framed as if it already happened.",
        "negation": "A competitor or option named only to be ruled out.",
        "hearsay_other_company": "Something said to have happened to a different, unnamed customer.",
        "generic_industry_talk": "Industry chatter with no account-specific claim.",
        "rhetorical_question": "A question asked for effect, not information, with no real open item.",
        "vendor_internal_speculation": "Vendor-side speculation about a customer, not a customer statement.",
        "resolved_issue_as_history": "An already-resolved issue retold as past history, not a live complaint.",
        "commitment_decision_confusable": (
            "A vendor commitment restated in decision-shaped language, or a customer "
            "decision restated as if it were still an open vendor promise -- easy to "
            "mistype as the other fact type."
        ),
    }
    out = []
    for cid, near_type in COMMON_DECOY_SHAPE:
        out.append(
            {
                "id": f"{cid}_{id_prefix}",
                "near_type": near_type,
                "description": descriptions[cid],
                "example": examples[cid],
            }
        )
    return out


def std_topic_tree(product_leaves: list[str]) -> list[dict]:
    return [
        {"name": "Product", "children": product_leaves},
        {"name": "Account", "children": ["Billing", "Renewal", "Expansion"]},
        {"name": "Support", "children": ["Incidents", "Training"]},
        {"name": "Competitive", "children": ["Evaluations"]},
        {"name": "People", "children": ["Contacts", "Champions"]},
    ]


# ============================================================================
# 1. senior_living_tech -- HARD GUARD: no fall detection/prevention, resident
#    monitoring, video monitoring, cameras, sensors, anywhere in products or
#    events. Care-plan documentation, medication management, staffing and
#    scheduling, family communication, billing, dietary, maintenance instead.
# ============================================================================
SENIOR_LIVING_TECH = {
    "id": "senior_living_tech",
    "name": "Senior-living tech",
    "guard_note": (
        "Guard (the maintainer, 2026-09-22): excludes resident fall detection, fall "
        "prevention, resident monitoring, video monitoring of residents, and any "
        "product in that category -- a category deliberately out of scope. Covers "
        "other senior-living operator tech only: care-plan documentation, "
        "medication management, staffing and scheduling, family communication, "
        "billing, dietary, maintenance. Enforced by "
        "`tests/test_world_generator_v2_packs.py::"
        "test_senior_living_pack_excludes_monitoring_category` (scans this pack's "
        "products and events for fall/monitor*/camera/video/sensor)."
    ),
    "vendor_what": (
        "care-plan documentation, medication management, staffing and scheduling, "
        "family communication, billing and dietary software sold to senior-living "
        "operators"
    ),
    "vocab": {
        "usage_unit": "first community",
        "staff_word": "care staff",
        "setup_friction": "care-plan documentation delays",
        "release_word": "eMAR update",
        "ops_role": "director of nursing",
        "incident_word": "medication-documentation error",
        "impact_area": "shift-handoff accuracy",
        "clinical_role": "director of nursing",
        "risk_topic": "care-plan accuracy",
        "usage_scope": "memory-care and assisted-living wings",
        "champion_role": "executive director",
        "praise_metric": "shift-handoff reliability",
        "integration_target": "state survey reporting integration",
        "product_noun": "care-management platform",
        "budget_role": "regional VP of operations",
        "expansion_scope": "second community",
        "expansion_area": "dietary and billing workflows",
        "remediation_word": "care-plan sync fix",
        "facilities_word": "communities",
        "policy_area": "medication-management",
    },
    "products": [
        {
            "kind": "care_plan_documentation",
            "name_template": "{line}",
            "tiers": [],
            "known_issue_pool": [
                "care-plan sync delays after a shift change",
                "PDF export formatting breaks on long care plans",
            ],
        },
        {
            "kind": "emar_medication_management",
            "name_template": "{line} {tier}",
            "tiers": ["MAR", "Plus"],
            "known_issue_pool": [
                "occasional duplicate dose alerts on PRN medications",
                "barcode scan failures on relabeled unit-dose packaging",
            ],
        },
        {
            "kind": "staffing_scheduling",
            "name_template": "{line} Schedule",
            "tiers": [],
            "known_issue_pool": [
                "shift-swap requests occasionally not routed to the right supervisor"
            ],
        },
        {
            "kind": "family_communication_portal",
            "name_template": "{line} Connect",
            "tiers": [],
            "known_issue_pool": [
                "photo uploads time out on the family app during peak evening hours"
            ],
        },
        {
            "kind": "billing_and_dietary",
            "name_template": "{line} Billing",
            "tiers": [],
            "known_issue_pool": [],
        },
        {
            "kind": "maintenance_work_orders",
            "name_template": "{line} Maintenance",
            "tiers": [],
            "known_issue_pool": ['work-order status occasionally stuck in "assigned"'],
        },
    ],
    "roles": [
        {"title": "Community Executive Director", "seniority": "director", "champion": True},
        {"title": "Director of Nursing", "seniority": "director", "champion": True},
        {"title": "Regional VP of Operations", "seniority": "vp", "champion": False},
        {"title": "Business Office Manager", "seniority": "manager", "champion": False},
        {"title": "Dietary Services Director", "seniority": "director", "champion": False},
        {"title": "Staffing Coordinator", "seniority": "manager", "champion": True},
        {"title": "Family Engagement Coordinator", "seniority": "staff", "champion": False},
        {"title": "IT Director", "seniority": "director", "champion": False},
    ],
    "topic_tree": std_topic_tree(
        ["Care plan documentation", "Medication management", "Staffing and scheduling"]
    ),
    "personas": personas("care_management", "sl"),
    "decoys": decoys(
        "sl",
        {
            "hypothetical": "if a care plan ever went unsigned for a shift, that would be a survey finding",
            "negation": "we're not looking at another care-management vendor right now",
            "hearsay_other_company": "I heard another community had the same eMAR sync issue",
            "generic_industry_talk": "staffing shortages in senior living have been in the trade press lately",
            "rhetorical_question": "who wouldn't want faster shift handoffs?",
            "vendor_internal_speculation": "our rep thinks the community might be shopping around at renewal",
            "resolved_issue_as_history": "we used to have care-plan sync delays, but that was fixed months ago",
            "commitment_decision_confusable": (
                "the exec director said the board already decided to add the dietary "
                "module, though the rep never actually confirmed a ship date"
            ),
        },
    ),
    "name_pool_refs": {
        "system": ["Senior Living Group", "Senior Communities", "Retirement Network"],
        "facility": [
            "Senior Living Community",
            "Assisted Living Community",
            "Memory Care Community",
        ],
        "standalone": ["Senior Care Partners", "Retirement Residence", "Life Plan Community"],
    },
    "excluded_product_kinds": [
        "resident_fall_detection",
        "resident_fall_prevention",
        "resident_monitoring",
        "video_monitoring",
    ],
}

# ============================================================================
# 2. pharmacy_services
# ============================================================================
PHARMACY_SERVICES = {
    "id": "pharmacy_services",
    "name": "Pharmacy services",
    "vendor_what": (
        "central-fill dispensing, medication synchronization, adherence packaging "
        "and specialty pharmacy software sold to health systems and pharmacies"
    ),
    "vocab": {
        "usage_unit": "first fill center",
        "staff_word": "pharmacy staff",
        "setup_friction": "workflow queue backlogs",
        "release_word": "dispensing-engine update",
        "ops_role": "pharmacy operations manager",
        "incident_word": "dispensing queue outage",
        "impact_area": "prescription fill turnaround",
        "clinical_role": "pharmacy director",
        "risk_topic": "fill-accuracy",
        "usage_scope": "central-fill and specialty lines",
        "champion_role": "pharmacy director",
        "praise_metric": "fill-turnaround time",
        "integration_target": "DIR-fee reporting integration",
        "product_noun": "dispensing platform",
        "budget_role": "chief pharmacy officer",
        "expansion_scope": "second fill center",
        "expansion_area": "adherence-packaging workflows",
        "remediation_word": "queue-backlog fix",
        "facilities_word": "pharmacy sites",
        "policy_area": "controlled-substance",
    },
    "products": [
        {
            "kind": "central_fill_dispensing",
            "name_template": "{line}",
            "tiers": [],
            "known_issue_pool": [
                "occasional canister miscounts after a refill cycle",
                "queue backlog during month-end volume spikes",
            ],
        },
        {
            "kind": "medication_synchronization",
            "name_template": "{line} Sync",
            "tiers": [],
            "known_issue_pool": [
                "sync-date reminders occasionally duplicate across household accounts"
            ],
        },
        {
            "kind": "adherence_packaging",
            "name_template": "{line} {tier}",
            "tiers": ["Pouch", "Blister"],
            "known_issue_pool": ["pouch-seal failures on high-humidity days"],
        },
        {
            "kind": "specialty_pharmacy_program",
            "name_template": "{line} Specialty",
            "tiers": [],
            "known_issue_pool": ["prior-authorization status sync lags the payer portal"],
        },
        {
            "kind": "pharmacy_inventory_automation",
            "name_template": "{line} Cabinet",
            "tiers": [],
            "known_issue_pool": [],
        },
    ],
    "roles": [
        {"title": "Pharmacy Director", "seniority": "director", "champion": True},
        {"title": "Chief Pharmacy Officer", "seniority": "c_suite", "champion": False},
        {"title": "Pharmacy Operations Manager", "seniority": "manager", "champion": True},
        {"title": "Clinical Pharmacist Lead", "seniority": "staff", "champion": True},
        {"title": "Procurement Manager", "seniority": "manager", "champion": False},
        {"title": "Compliance Officer", "seniority": "manager", "champion": False},
        {"title": "CFO", "seniority": "c_suite", "champion": False},
        {"title": "Inventory Manager", "seniority": "staff", "champion": False},
    ],
    "topic_tree": std_topic_tree(
        ["Central-fill dispensing", "Medication synchronization", "Adherence packaging"]
    ),
    "personas": personas("pharmacy_ops", "ph"),
    "decoys": decoys(
        "ph",
        {
            "hypothetical": "if a canister ever miscounted twice in a week, that would trigger an audit",
            "negation": "we're not evaluating another dispensing vendor at the moment",
            "hearsay_other_company": "I heard another pharmacy group had the same sync-date issue",
            "generic_industry_talk": "DIR fee reporting rules have been in the pharmacy trade press lately",
            "rhetorical_question": "who wouldn't want fewer canister miscounts?",
            "vendor_internal_speculation": "our rep thinks the account might be shopping around at renewal",
            "resolved_issue_as_history": "we used to have queue backlogs, but that was fixed months ago",
            "commitment_decision_confusable": (
                "the ops manager said they've already decided to add adherence packaging, "
                "though the rep never actually confirmed a rollout date"
            ),
        },
    ),
    "name_pool_refs": {
        "system": ["Pharmacy Health System", "Regional Pharmacy Network", "Health Pharmacy Group"],
        "facility": ["Community Pharmacy", "Specialty Pharmacy Center", "Infusion Pharmacy"],
        "standalone": ["Pharmacy Partners", "Pharmacy Care Group", "Compounding Pharmacy"],
    },
    "excluded_product_kinds": [],
}

# ============================================================================
# 3. ehr_addons
# ============================================================================
EHR_ADDONS = {
    "id": "ehr_addons",
    "name": "EHR add-ons",
    "vendor_what": (
        "clinical documentation, interoperability, e-prescribing and revenue-cycle "
        "add-on modules sold to health systems running third-party EHRs"
    ),
    "vocab": {
        "usage_unit": "first service line",
        "staff_word": "clinical informatics staff",
        "setup_friction": "template-mapping delays",
        "release_word": "interface-engine update",
        "ops_role": "IT integration manager",
        "incident_word": "HL7 interface outage",
        "impact_area": "clinical documentation workflow",
        "clinical_role": "CMIO",
        "risk_topic": "data-mapping accuracy",
        "usage_scope": "inpatient and ambulatory modules",
        "champion_role": "CMIO",
        "praise_metric": "documentation turnaround",
        "integration_target": "population-health dashboard integration",
        "product_noun": "interoperability module",
        "budget_role": "CIO",
        "expansion_scope": "second service line",
        "expansion_area": "revenue-cycle coding workflows",
        "remediation_word": "interface-mapping fix",
        "facilities_word": "service lines",
        "policy_area": "clinical-documentation",
    },
    "products": [
        {
            "kind": "clinical_documentation_improvement",
            "name_template": "{line}",
            "tiers": [],
            "known_issue_pool": [
                "template auto-save lag on long encounter notes",
                "coding-suggestion mismatches on complex encounters",
            ],
        },
        {
            "kind": "interoperability_engine",
            "name_template": "{line} Connect",
            "tiers": [],
            "known_issue_pool": ["HL7 message queue delays during EHR maintenance windows"],
        },
        {
            "kind": "e_prescribing_addon",
            "name_template": "{line} {tier}",
            "tiers": ["Rx", "RxPlus"],
            "known_issue_pool": ["formulary lookup timeouts during payer outages"],
        },
        {
            "kind": "population_health_dashboard",
            "name_template": "{line} Insights",
            "tiers": [],
            "known_issue_pool": [],
        },
        {
            "kind": "revenue_cycle_coding_assist",
            "name_template": "{line} Code",
            "tiers": [],
            "known_issue_pool": [
                "coding suggestions occasionally lag a payer's updated fee schedule"
            ],
        },
    ],
    "roles": [
        {"title": "CIO", "seniority": "c_suite", "champion": True},
        {"title": "CMIO", "seniority": "c_suite", "champion": True},
        {
            "title": "Director of Health Information Management",
            "seniority": "director",
            "champion": True,
        },
        {"title": "IT Integration Manager", "seniority": "manager", "champion": False},
        {"title": "VP Revenue Cycle", "seniority": "vp", "champion": False},
        {"title": "Physician Champion", "seniority": "staff", "champion": True},
        {"title": "Compliance Officer", "seniority": "manager", "champion": False},
        {"title": "CFO", "seniority": "c_suite", "champion": False},
    ],
    "topic_tree": std_topic_tree(
        ["Clinical documentation", "Interoperability", "Revenue-cycle coding"]
    ),
    "personas": personas("health_it", "ehr"),
    "decoys": decoys(
        "ehr",
        {
            "hypothetical": "if the interface engine ever dropped a message silently, that would be a patient-safety issue",
            "negation": "we're not evaluating another interoperability vendor right now",
            "hearsay_other_company": "I heard another health system had the same HL7 queue delay",
            "generic_industry_talk": "interoperability mandates have been in the health-IT trade press lately",
            "rhetorical_question": "who wouldn't want faster documentation turnaround?",
            "vendor_internal_speculation": "our rep thinks the account might be evaluating a competing module",
            "resolved_issue_as_history": "we used to have HL7 queue delays, but that was fixed months ago",
            "commitment_decision_confusable": (
                "the CMIO said the board already decided to expand the coding-assist module, "
                "though the rep never actually confirmed a go-live date"
            ),
        },
    ),
    "name_pool_refs": {
        "system": [
            "Health Informatics Network",
            "Integrated Health System",
            "Clinical Data Network",
        ],
        "facility": ["Ambulatory Care Center", "Outpatient Clinic", "Specialty Clinic"],
        "standalone": ["Medical Group", "Physician Partners", "Clinical Associates"],
    },
    "excluded_product_kinds": [],
}

# ============================================================================
# 4. healthcare_staffing
# ============================================================================
HEALTHCARE_STAFFING = {
    "id": "healthcare_staffing",
    "name": "Healthcare staffing",
    "vendor_what": (
        "nurse staffing marketplace, credentialing and vendor-management software "
        "sold to hospitals and health systems"
    ),
    "vocab": {
        "usage_unit": "first unit",
        "staff_word": "the staffing office",
        "setup_friction": "credential-verification delays",
        "release_word": "shift-marketplace update",
        "ops_role": "staffing coordinator",
        "incident_word": "shift-fill outage",
        "impact_area": "shift coverage",
        "clinical_role": "chief nursing officer",
        "risk_topic": "fill-rate",
        "usage_scope": "float pool and contingent labor programs",
        "champion_role": "chief nursing officer",
        "praise_metric": "fill rate",
        "integration_target": "payroll system integration",
        "product_noun": "staffing marketplace",
        "budget_role": "VP workforce management",
        "expansion_scope": "second facility",
        "expansion_area": "credentialing workflows",
        "remediation_word": "credential-sync fix",
        "facilities_word": "facilities",
        "policy_area": "credentialing",
    },
    "products": [
        {
            "kind": "nurse_staffing_marketplace",
            "name_template": "{line}",
            "tiers": [],
            "known_issue_pool": [
                "shift-bid notifications occasionally delayed during peak posting windows",
                "duplicate shift postings after a schedule edit",
            ],
        },
        {
            "kind": "credentialing_management",
            "name_template": "{line} Credential",
            "tiers": [],
            "known_issue_pool": ["license-expiration alerts occasionally fire a day late"],
        },
        {
            "kind": "vendor_management_system",
            "name_template": "{line} {tier}",
            "tiers": ["VMS", "VMS Pro"],
            "known_issue_pool": ["rate-card mismatches between agencies during contract renewals"],
        },
        {
            "kind": "float_pool_scheduling",
            "name_template": "{line} Float",
            "tiers": [],
            "known_issue_pool": [],
        },
        {
            "kind": "compliance_tracking",
            "name_template": "{line} Compliance",
            "tiers": [],
            "known_issue_pool": ["compliance dashboard occasionally lags nightly credential sync"],
        },
    ],
    "roles": [
        {"title": "Chief Nursing Officer", "seniority": "c_suite", "champion": True},
        {"title": "Staffing Coordinator", "seniority": "manager", "champion": True},
        {"title": "HR/Talent Acquisition Director", "seniority": "director", "champion": False},
        {"title": "VP Workforce Management", "seniority": "vp", "champion": True},
        {"title": "Director of Clinical Operations", "seniority": "director", "champion": False},
        {"title": "CFO", "seniority": "c_suite", "champion": False},
        {"title": "Credentialing Manager", "seniority": "manager", "champion": False},
        {"title": "Nurse Manager", "seniority": "manager", "champion": True},
    ],
    "topic_tree": std_topic_tree(["Shift marketplace", "Credentialing", "Vendor management"]),
    "personas": personas("workforce_ops", "st"),
    "decoys": decoys(
        "st",
        {
            "hypothetical": "if a shift ever went unfilled past the SLA window, that would show up in the fill-rate report",
            "negation": "we're not looking at another staffing marketplace right now",
            "hearsay_other_company": "I heard another health system had the same credential-alert delay",
            "generic_industry_talk": "travel-nurse rate volatility has been in the staffing trade press lately",
            "rhetorical_question": "who wouldn't want a higher fill rate?",
            "vendor_internal_speculation": "our rep thinks the account might be consolidating vendors at renewal",
            "resolved_issue_as_history": "we used to have late credential alerts, but that was fixed months ago",
            "commitment_decision_confusable": (
                "the CNO said they've already decided to expand the float-pool module, "
                "though the rep never actually confirmed a rollout date"
            ),
        },
    ),
    "name_pool_refs": {
        "system": [
            "Staffing Health System",
            "Workforce Health Network",
            "Regional Staffing Alliance",
        ],
        "facility": ["Acute Care Hospital", "Skilled Nursing Facility", "Rehabilitation Hospital"],
        "standalone": ["Staffing Partners", "Clinical Workforce Group", "Locum Partners"],
    },
    "excluded_product_kinds": [],
}

# ============================================================================
# 5. medical_supplies
# ============================================================================
MEDICAL_SUPPLIES = {
    "id": "medical_supplies",
    "name": "Medical supplies",
    "vendor_what": (
        "consumable medical supplies, par-level inventory management and "
        "supply-chain analytics software sold to hospitals and surgery centers"
    ),
    "vocab": {
        "usage_unit": "first unit",
        "staff_word": "materials staff",
        "setup_friction": "par-level setup delays",
        "release_word": "inventory-system update",
        "ops_role": "materials manager",
        "incident_word": "supply stockout",
        "impact_area": "supply availability",
        "clinical_role": "director of perioperative services",
        "risk_topic": "fill-rate",
        "usage_scope": "OR and central-supply units",
        "champion_role": "materials manager",
        "praise_metric": "on-time fill rate",
        "integration_target": "GPO contract-management integration",
        "product_noun": "supply-chain platform",
        "budget_role": "VP supply chain",
        "expansion_scope": "second facility",
        "expansion_area": "surgical-kitting workflows",
        "remediation_word": "restock plan",
        "facilities_word": "facilities",
        "policy_area": "procurement",
    },
    "products": [
        {
            "kind": "wound_care_supplies",
            "name_template": "{line}",
            "tiers": [],
            "known_issue_pool": [
                "backorders on specialty dressing sizes during high-demand periods",
                "lot-tracking mismatches after a supplier substitution",
            ],
        },
        {
            "kind": "par_level_inventory_system",
            "name_template": "{line} Par",
            "tiers": [],
            "known_issue_pool": ["par-count drift after a manual cycle count"],
        },
        {
            "kind": "surgical_kitting",
            "name_template": "{line} {tier}",
            "tiers": ["Kit", "KitPlus"],
            "known_issue_pool": ["kit-pick errors on custom procedure trays"],
        },
        {
            "kind": "supply_chain_analytics",
            "name_template": "{line} Insights",
            "tiers": [],
            "known_issue_pool": [],
        },
        {
            "kind": "gpo_contract_management",
            "name_template": "{line} Contracts",
            "tiers": [],
            "known_issue_pool": ["contract-tier pricing occasionally lags a GPO amendment"],
        },
    ],
    "roles": [
        {"title": "Materials Manager", "seniority": "manager", "champion": True},
        {"title": "VP Supply Chain", "seniority": "vp", "champion": True},
        {"title": "Director of Perioperative Services", "seniority": "director", "champion": True},
        {"title": "CFO", "seniority": "c_suite", "champion": False},
        {"title": "Central Sterile Processing Manager", "seniority": "manager", "champion": False},
        {"title": "Procurement Director", "seniority": "director", "champion": False},
        {"title": "Supply Chain Analyst", "seniority": "staff", "champion": False},
        {"title": "COO", "seniority": "c_suite", "champion": False},
    ],
    "topic_tree": std_topic_tree(
        ["Wound care supplies", "Par-level inventory", "Surgical kitting"]
    ),
    "personas": personas("supply_chain", "ms"),
    "decoys": decoys(
        "ms",
        {
            "hypothetical": "if a par-level stockout ever hit the OR mid-case, that would be a real problem",
            "negation": "we're not evaluating another supply-chain vendor right now",
            "hearsay_other_company": "I heard another facility had the same par-count drift issue",
            "generic_industry_talk": "GPO contract consolidation has been in the supply-chain trade press lately",
            "rhetorical_question": "who wouldn't want a higher on-time fill rate?",
            "vendor_internal_speculation": "our rep thinks the account might be re-bidding the contract",
            "resolved_issue_as_history": "we used to have par-count drift, but that was fixed months ago",
            "commitment_decision_confusable": (
                "the materials manager said they've already decided to expand surgical "
                "kitting, though the rep never actually confirmed a rollout date"
            ),
        },
    ),
    "name_pool_refs": {
        "system": [
            "Supply Chain Health System",
            "Regional Materials Network",
            "Health Materials Group",
        ],
        "facility": ["Surgery Center", "Acute Care Center", "Outpatient Surgery Center"],
        "standalone": ["Supply Partners", "Materials Group", "Procurement Alliance"],
    },
    "excluded_product_kinds": [],
}

# ============================================================================
# 6. telehealth
# ============================================================================
TELEHEALTH = {
    "id": "telehealth",
    "name": "Telehealth",
    "vendor_what": (
        "video-visit, remote patient monitoring and virtual behavioral health "
        "platforms sold to health systems and payers"
    ),
    "vocab": {
        "usage_unit": "first service line",
        "staff_word": "the care team",
        "setup_friction": "visit-scheduling delays",
        "release_word": "video-platform update",
        "ops_role": "telehealth program lead",
        "incident_word": "video-visit outage",
        "impact_area": "virtual-visit access",
        "clinical_role": "chief medical officer",
        "risk_topic": "visit-completion",
        "usage_scope": "primary-care and behavioral-health service lines",
        "champion_role": "telehealth program lead",
        "praise_metric": "visit-completion rate",
        "integration_target": "remote-monitoring device integration",
        "product_noun": "virtual-care platform",
        "budget_role": "VP ambulatory services",
        "expansion_scope": "second service line",
        "expansion_area": "behavioral-health workflows",
        "remediation_word": "video-quality fix",
        "facilities_word": "service lines",
        "policy_area": "virtual-visit",
    },
    "products": [
        {
            "kind": "video_visit_platform",
            "name_template": "{line}",
            "tiers": [],
            "known_issue_pool": [
                "video-quality drops on low-bandwidth patient connections",
                "waiting-room notifications occasionally delayed",
            ],
        },
        {
            "kind": "remote_patient_monitoring",
            "name_template": "{line} RPM",
            "tiers": [],
            "known_issue_pool": ["device-reading sync lags for cellular-connected devices"],
        },
        {
            "kind": "async_econsult_module",
            "name_template": "{line} {tier}",
            "tiers": ["Consult", "ConsultPlus"],
            "known_issue_pool": ["attachment upload failures on large imaging files"],
        },
        {
            "kind": "virtual_behavioral_health",
            "name_template": "{line} Behavioral",
            "tiers": [],
            "known_issue_pool": [],
        },
        {
            "kind": "patient_engagement_app",
            "name_template": "{line} Engage",
            "tiers": [],
            "known_issue_pool": ["appointment reminders occasionally duplicate across channels"],
        },
    ],
    "roles": [
        {"title": "Telehealth Program Lead", "seniority": "manager", "champion": True},
        {"title": "Chief Medical Officer", "seniority": "c_suite", "champion": True},
        {"title": "VP Ambulatory Services", "seniority": "vp", "champion": True},
        {"title": "IT Director", "seniority": "director", "champion": False},
        {"title": "Population Health Director", "seniority": "director", "champion": False},
        {"title": "CFO", "seniority": "c_suite", "champion": False},
        {"title": "Nurse Practitioner Lead", "seniority": "staff", "champion": False},
        {"title": "Behavioral Health Director", "seniority": "director", "champion": False},
    ],
    "topic_tree": std_topic_tree(
        ["Video visits", "Remote patient monitoring", "Virtual behavioral health"]
    ),
    "personas": personas("telehealth_clinical", "th"),
    "decoys": decoys(
        "th",
        {
            "hypothetical": "if a video visit ever dropped mid-consult, that would hurt the completion rate",
            "negation": "we're not evaluating another virtual-care vendor right now",
            "hearsay_other_company": "I heard another health system had the same video-quality issue",
            "generic_industry_talk": "telehealth reimbursement policy has been in the trade press lately",
            "rhetorical_question": "who wouldn't want a higher visit-completion rate?",
            "vendor_internal_speculation": "our rep thinks the account might be evaluating a competing platform",
            "resolved_issue_as_history": "we used to have video-quality drops, but that was fixed months ago",
            "commitment_decision_confusable": (
                "the program lead said they've already decided to expand the behavioral-"
                "health module, though the rep never actually confirmed a go-live date"
            ),
        },
    ),
    "name_pool_refs": {
        "system": ["Virtual Health Network", "Telehealth Health System", "Connected Care Network"],
        "facility": ["Telehealth Clinic", "Virtual Care Center", "Ambulatory Telehealth Site"],
        "standalone": ["Telehealth Partners", "Virtual Care Group", "Connected Health Partners"],
    },
    "excluded_product_kinds": [],
}

# ============================================================================
# 7. lab_and_imaging_services -- HELD OUT (test worlds only). Spec section 4
#    names this verbatim as its own example of the eighth held-out vertical:
#    "an eighth held-out vertical (for example lab and imaging services)".
# ============================================================================
LAB_AND_IMAGING_SERVICES = {
    "id": "lab_and_imaging_services",
    "name": "Lab and imaging services",
    "held_out": True,
    "vendor_what": (
        "laboratory information systems, radiology workflow, PACS image archiving "
        "and results-reporting software sold to health systems, imaging centers "
        "and independent labs"
    ),
    "vocab": {
        "usage_unit": "first site",
        "staff_word": "lab staff",
        "setup_friction": "specimen-routing delays",
        "release_word": "LIS/RIS update",
        "ops_role": "lab operations manager",
        "incident_word": "results-delivery outage",
        "impact_area": "turnaround time",
        "clinical_role": "radiology medical director",
        "risk_topic": "turnaround-time",
        "usage_scope": "core lab and imaging reading pools",
        "champion_role": "lab director",
        "praise_metric": "turnaround time",
        "integration_target": "EHR results-integration",
        "product_noun": "diagnostics platform",
        "budget_role": "VP diagnostic services",
        "expansion_scope": "second site",
        "expansion_area": "specimen-routing workflows",
        "remediation_word": "results-delivery fix",
        "facilities_word": "sites",
        "policy_area": "specimen-handling",
    },
    "products": [
        {
            "kind": "laboratory_information_system",
            "name_template": "{line}",
            "tiers": [],
            "known_issue_pool": [
                "specimen-routing mismatches during high-volume days",
                "results-delivery lag during batch processing",
            ],
        },
        {
            "kind": "radiology_information_system",
            "name_template": "{line} RIS",
            "tiers": [],
            "known_issue_pool": ["worklist sync delays with the reading-room PACS"],
        },
        {
            "kind": "pacs_image_archive",
            "name_template": "{line} {tier}",
            "tiers": ["Archive", "ArchivePlus"],
            "known_issue_pool": ["large-study upload timeouts on legacy scanners"],
        },
        {
            "kind": "specimen_tracking",
            "name_template": "{line} Track",
            "tiers": [],
            "known_issue_pool": [],
        },
        {
            "kind": "results_reporting_portal",
            "name_template": "{line} Report",
            "tiers": [],
            "known_issue_pool": ["critical-result alerts occasionally delayed past the SLA window"],
        },
    ],
    "roles": [
        {"title": "Lab Director", "seniority": "director", "champion": True},
        {"title": "Radiology Medical Director", "seniority": "c_suite", "champion": True},
        {"title": "VP Diagnostic Services", "seniority": "vp", "champion": True},
        {"title": "Lab Operations Manager", "seniority": "manager", "champion": False},
        {"title": "Pathologist Lead", "seniority": "staff", "champion": False},
        {"title": "Compliance Officer", "seniority": "manager", "champion": False},
        {"title": "CFO", "seniority": "c_suite", "champion": False},
        {"title": "Referring Physician Liaison", "seniority": "staff", "champion": False},
    ],
    "topic_tree": std_topic_tree(
        ["Laboratory information system", "Radiology workflow", "Results reporting"]
    ),
    "personas": personas("diagnostics_ops", "li"),
    "decoys": decoys(
        "li",
        {
            "hypothetical": "if a critical result ever missed the SLA window, that would trigger an incident review",
            "negation": "we're not evaluating another LIS vendor right now",
            "hearsay_other_company": "I heard another lab network had the same worklist sync issue",
            "generic_industry_talk": "reference-lab consolidation has been in the diagnostics trade press lately",
            "rhetorical_question": "who wouldn't want a faster turnaround time?",
            "vendor_internal_speculation": "our rep thinks the account might be evaluating a competing LIS",
            "resolved_issue_as_history": "we used to have worklist sync delays, but that was fixed months ago",
            "commitment_decision_confusable": (
                "the lab director said the board already decided to expand the PACS "
                "archive, though the rep never actually confirmed a migration date"
            ),
        },
    ),
    "name_pool_refs": {
        "system": ["Diagnostics Health System", "Regional Lab Network", "Imaging Health Network"],
        "facility": ["Diagnostic Imaging Center", "Reference Laboratory", "Radiology Center"],
        "standalone": ["Diagnostics Partners", "Pathology Group", "Imaging Associates"],
    },
    "excluded_product_kinds": [],
}

ALL_VERTICALS = [
    SENIOR_LIVING_TECH,
    PHARMACY_SERVICES,
    EHR_ADDONS,
    HEALTHCARE_STAFFING,
    MEDICAL_SUPPLIES,
    TELEHEALTH,
    LAB_AND_IMAGING_SERVICES,
]
