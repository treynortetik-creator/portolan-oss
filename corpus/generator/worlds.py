"""Generate a fresh, fully-synthetic truth table from a seed.

`corpus/truth-table.yaml` (hand-written 2026-09-12/13) is the ANSWER KEY for
`corpus/out/` and stays exactly as it is -- `python -m corpus.generator.cli
generate` still reads only that file. This module produces a NEW truth
table, in the same YAML shape, so `briefs.py`'s existing
`build_calls`/`build_tickets`/`build_emails`/`build_crm` consume it
unchanged (build instructions item 1: "Output the same YAML shape as
corpus/truth-table.yaml so briefs.py consumes it unchanged"). Slack, Teams,
and the richer CRM records (opportunities/cases/activity notes) are new
source kinds `briefs.py` never had, so this module also emits two sections
(`sources.slack`, `sources.teams`, `sources.crm_notes`) that
`briefs.build_chat_briefs`/`build_crm_notes` (added alongside, in
`briefs.py`) read directly -- see that module's docstrings.

Everything here is fictional. Every account, contact, internal rep, and
branch name is freshly generated per seed from name banks disjoint from
the ones in `corpus/truth-table.yaml`, so a synth batch never reads like
the hand-built corpus with the serial numbers filed off.

**Vendor mythology (added 2026-09-21, `01-spec/synthetic-data-pipeline-
2026-09-19.md` "distinct worlds"):** the vendor identity itself -- supplier
name/domain, the 5-product line, and the 2 named competitors -- is now ALSO
picked fresh per seed from `VENDOR_BANK`/`PRODUCT_LINE_NAMES`/
`COMPETITOR_BANK` below, never "Halvard Medical"/"Tidewater"/"Corvane"/
"Brightline" (the hand-built corpus's fixed cast): a model trained on many
synth batches must not learn "the vendor is always Halvard Medical" as
part of the extraction task. This was previously kept fixed for a reason
that turned out not to apply: `manifest.LOCATOR_PATTERNS["email"]`
validates the EMAIL LOCATOR (`<source_id>@<domain>#p<n>`, built from
`structured.MESSAGE_ID_DOMAIN`, a technical envelope constant, always
`halvardmedical.com`, independent of any person's own address) -- not
`meta.supplier.domain` or any contact/internal-person `email` field, which
were always free to vary. `structured.MESSAGE_ID_DOMAIN` itself stays
fixed (a Message-ID host suffix, not prose a model reads as "who is the
vendor" -- see that module).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

# --------------------------------------------------------------------------
# Vendor mythology banks -- seed-picked per world (see this module's
# docstring). Every name here is disjoint from the hand-built corpus's
# fixed cast ("Halvard Medical", "Tidewater", "Corvane Systems",
# "Brightline Medical").
# --------------------------------------------------------------------------

VENDOR_BANK: list[dict[str, str]] = [
    {"name": "Meridian Infusion Systems", "short": "Meridian"},
    {"name": "Solace Medical", "short": "Solace"},
    {"name": "Northlight Health Devices", "short": "Northlight"},
    {"name": "Anchorpoint Medical", "short": "Anchorpoint"},
    {"name": "Cascadia Infusion", "short": "Cascadia"},
    {"name": "Vantage Biomedical", "short": "Vantage"},
    {"name": "Fernbridge Medical", "short": "Fernbridge"},
    {"name": "Ashworth Infusion Technologies", "short": "Ashworth"},
]

PRODUCT_LINE_NAMES: list[str] = [
    "Current",
    "Meraline",
    "Streamline",
    "Corewell",
    "Trueflow",
    "Beacon",
    "Farview",
    "Alderline",
    "Northstar",
    "Clearpath",
]

COMPETITOR_BANK: list[dict[str, str]] = [
    {"id": "westhaven", "name": "Westhaven Devices", "aliases": ["Westhaven"]},
    {"id": "palisade", "name": "Palisade Medical", "aliases": ["Palisade"]},
    {"id": "orchard-health", "name": "Orchard Health Systems", "aliases": ["Orchard"]},
    {"id": "thornfield", "name": "Thornfield Medical", "aliases": ["Thornfield"]},
    {"id": "grayling", "name": "Grayling Devices", "aliases": ["Grayling"]},
    {"id": "ambervale", "name": "Ambervale Medical", "aliases": ["Ambervale"]},
]


def _pick_vendor_mythology(
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """One vendor identity for this world: a supplier (name/short/domain/
    what), its 5-entry product line (family, T2, T3, Link, service
    contract -- same shape `_make_fact`'s templates and `generate_world`'s
    `meta` block expect), and 2 competitors -- freshly picked from the
    banks above. Called once per world, early in `generate_world`, so it
    consumes the front of that world's `rng` stream."""
    vendor = rng.choice(VENDOR_BANK)
    domain = f"{_slug(vendor['short'])}.com"
    supplier = {
        "name": vendor["name"],
        "short": vendor["short"],
        "domain": domain,
        "what": "infusion pumps and integration modules sold to hospitals and health systems",
    }
    line = rng.choice(PRODUCT_LINE_NAMES)
    care_name = f"{vendor['short']}Care"
    products = [
        {
            "id": "product-line",
            "name": line,
            "kind": "product_family",
            "aliases": [f"{line} consumables", f"{line} pumps", f"{line} line"],
        },
        {"id": "product-t2", "name": f"{line} T2", "aliases": ["T2", "the T2", f"{line} pump"]},
        {"id": "product-t3", "name": f"{line} T3", "aliases": ["T3", "the new pump", "T3 pump"]},
        {
            "id": "product-link",
            "name": f"{line} Link",
            "aliases": ["Link", "the Link module", "EHR bridge"],
        },
        {
            "id": "service-contract",
            "name": f"{care_name} service contract",
            "aliases": [care_name, "the service plan"],
        },
    ]
    competitors = rng.sample(COMPETITOR_BANK, 2)
    return supplier, products, competitors


# --------------------------------------------------------------------------
# Name banks -- disjoint from every account/contact/internal-person name in
# corpus/truth-table.yaml (checked by hand 2026-09-19), so a generated
# world is never confusable with the hand-built corpus.
#
# `ACCOUNT_STEMS`/`FIRST_NAMES` are large enough (2026-09-21, "distinct
# worlds" -- 01-spec/synthetic-data-pipeline-2026-09-19.md) that two
# different seeds drawing a small `--accounts`/contact count essentially
# never land on the exact same stem/first-name set -- `_HAND_CORPUS_STEMS`/
# `_HAND_CORPUS_FIRST_NAMES` below are excluded by construction so neither
# bank can ever exactly reproduce a hand-built-corpus name either.
# --------------------------------------------------------------------------

_HAND_CORPUS_STEMS = frozenset(
    {
        "pinecrest",
        "larkspur",
        "ashgrove",
        "cobalt ridge",
        "northlake",
        "meadowbrook",
        "st. aldric",
        "vireo",
        "sable valley",
        "kestrel",
        "wrenfield",
        "driftwood",
        "anchor bay",
    }
)

_ORIGINAL_ACCOUNT_STEMS = [
    "Briarcliff",
    "Hollowbrook",
    "Tanglewood",
    "Cedar Hollow",
    "Marrow Creek",
    "Windmere",
    "Thistlewood",
    "Harborview",
    "Stonebridge",
    "Fallowfield",
    "Ridgeline",
    "Copper Bend",
    "Silverlake",
    "Amberton",
    "Foxglen",
    "Brackenfield",
    "Moss Landing",
    "Quill Harbor",
    "Redstone",
    "Elderwood",
    "Graymoor",
    "Thornbury",
    "Cinderpine",
    "Wolverton",
    "Hazelmere",
    "Lumenfield",
    "Windcrest",
    "Saltmarsh",
    "Ironvale",
    "Millbrook",
    "Cloverdale",
    "Bramblewood",
    "Oakhaven",
    "Pemberton",
    "Rosewick",
    "Barrowfield",
]

# Composed (not hand-typed one by one) so the bank can be large enough for
# seed diversity without hundreds of literal strings to eyeball -- every
# combo landing on a hand-corpus stem (`_HAND_CORPUS_STEMS`) is dropped.
_STEM_PREFIXES = [
    "Briar", "Hollow", "Tangle", "Marrow", "Wind", "Thistle", "Harbor",
    "Stone", "Fallow", "Ridge", "Copper", "Silver", "Amber", "Fox",
    "Bracken", "Moss", "Quill", "Red", "Elder", "Gray", "Thorn", "Cinder",
    "Wolver", "Hazel", "Lumen", "Salt", "Iron", "Mill", "Clover",
    "Bramble", "Oak", "Pember", "Rose", "Barrow", "Fern", "Birch", "Maple",
    "Alder", "Willow", "Heather",
]  # fmt: skip
_STEM_SUFFIXES = [
    "cliff", "brook", "wood", "creek", "mere", "field", "view", "bridge",
    "ridge", "bend", "lake", "ton", "glen", "bury", "pine", "mont",
    "harbor", "stone", "haven", "hollow",
]  # fmt: skip

ACCOUNT_STEMS: list[str] = list(_ORIGINAL_ACCOUNT_STEMS)
for _prefix in _STEM_PREFIXES:
    for _suffix in _STEM_SUFFIXES:
        _combo = f"{_prefix}{_suffix}"
        if _combo.lower() not in _HAND_CORPUS_STEMS and _combo not in ACCOUNT_STEMS:
            ACCOUNT_STEMS.append(_combo)
del _prefix, _suffix, _combo

SYSTEM_SUFFIXES = ["Health", "Regional Health", "Health System"]
FACILITY_SUFFIXES = ["Medical Center", "Community Hospital", "General Hospital"]
STANDALONE_SUFFIXES = ["Surgical Group", "Surgical Partners", "Health Partners"]

REGIONS = ["Midwest", "Northeast", "South", "West"]

# 26 original + 60 more (2026-09-21, "distinct worlds") -- disjoint from
# each other, from the hand-built corpus's contact/internal-person first
# names, and from LAST_NAMES/ACCOUNT_STEMS/the vendor banks above.
FIRST_NAMES = [
    "Alina",
    "Bennett",
    "Cora",
    "Desmond",
    "Elowen",
    "Farid",
    "Greta",
    "Hollis",
    "Imani",
    "Jasper",
    "Keiko",
    "Lior",
    "Maren",
    "Nadir",
    "Odessa",
    "Pieter",
    "Quinby",
    "Rosalind",
    "Silas",
    "Tamsin",
    "Ulric",
    "Vesna",
    "Wren",
    "Xander",
    "Yara",
    "Zeke",
    "Amara",
    "Bastian",
    "Calla",
    "Dashiell",
    "Eamon",
    "Fenwick",
    "Gisela",
    "Henrik",
    "Ines",
    "Jolene",
    "Kato",
    "Liora",
    "Magnus",
    "Neve",
    "Osric",
    "Petra",
    "Quen",
    "Rune",
    "Saskia",
    "Tobias",
    "Ursula",
    "Vance",
    "Wynn",
    "Xiomara",
    "Yusuf",
    "Zola",
    "Anouk",
    "Baxter",
    "Celeste",
    "Dorian",
    "Elian",
    "Freya",
    "Gideon",
    "Harlan",
    "Ivette",
    "Kellan",
    "Lennox",
    "Mireille",
    "Niamh",
    "Orin",
    "Percy",
    "Quilla",
    "Rosalyn",
    "Soren",
    "Talise",
    "Ulla",
    "Vaughn",
    "Wilhelm",
    "Xanthe",
    "Ysolde",
    "Zinnia",
    "Anders",
    "Briony",
    "Castor",
    "Deirdre",
]

LAST_NAMES = [
    "Abernathy",
    "Brandt",
    "Calloway",
    "Deveraux",
    "Ekstrom",
    "Farrow",
    "Gable",
    "Hartigan",
    "Ibsen",
    "Jareau",
    "Kestner",
    "Lindqvist",
    "Marchetti",
    "Nakamura",
    "Ostrander",
    "Prewitt",
    "Quintana",
    "Renfield",
    "Sorensen",
    "Tillman",
    "Uribe",
    "Voss",
    "Whitfield",
    "Yaeger",
    "Zamora",
    "Anholt",
    "Bellamy",
    "Carrow",
    "Delacroix",
    "Eastwood",
    "Fenn",
    "Grantham",
    "Holloway",
    "Isaacs",
    "Jourdain",
    "Kingsley",
    "Larsen",
    "Mercer",
    "Norwood",
    "Osgood",
    "Pennington",
    "Quinlan",
    "Rutherford",
    "Stavros",
    "Thackeray",
    "Vasquez",
    "Wexford",
    "Yardley",
    "Zimmerman",
    "Ashby",
    "Brennan",
    "Carrick",
    "Dunmore",
    "Ellery",
    "Fairweather",
    "Gantry",
    "Holt",
]

CONTACT_TITLES = [
    "Director of Procurement",
    "VP Supply Chain",
    "Biomed Engineering Lead",
    "Materials Manager",
    "CFO",
    "Practice Administrator",
    "Director of Clinical Engineering",
    "Pharmacy Director",
    "VP Operations",
    "Director of Nursing Informatics",
    "ICU Nurse Manager",
    "Director of Pharmacy",
]

INTERNAL_ROLE_POOL = [
    ("account rep", "Midwest"),
    ("account rep", "Northeast"),
    ("customer success lead", None),
    ("operations manager", None),
    ("product manager", None),
]

BRANCH_NAMES = ["Dayton", "Columbus", "Hartford", "Tulsa", "Reno"]

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

# Realistic proportions (build instructions item 1) -- late-delivery
# complaints and requests dominate a real support/sales stream; contact
# changes, risk flags, and "other" (internal ops chatter) are rarer.
FACT_TYPE_WEIGHTS: dict[str, float] = {
    "complaint": 0.20,
    "request": 0.14,
    "praise": 0.09,
    "risk": 0.05,
    "competitor_mention": 0.10,
    "commitment": 0.08,
    "decision": 0.08,
    "contact_change": 0.05,
    "pricing": 0.10,
    "usage": 0.06,
    "question": 0.03,
    "other": 0.02,
}

# Fact types that support a later "resolution"/reversal fact superseding an
# earlier one (mirrors F11->F11b, F15->F15b in the hand-built corpus).
SUPERSEDABLE_TYPES = {"complaint", "competitor_mention", "decision", "pricing", "risk"}


@dataclass
class WorldAccount:
    id: str
    crm_id: str
    name: str
    kind: str
    parent: str | None
    domain: str
    arr: int
    region: str
    aliases: list[str]


@dataclass
class WorldContact:
    id: str
    crm_id: str
    name: str
    account: str
    title: str
    email: str


@dataclass
class WorldPerson:
    id: str
    name: str
    role: str
    territory: str | None
    email: str


@dataclass
class WorldFact:
    id: str
    type: str
    claim: str
    entities: list[str]
    date: str
    speaker: str
    kind: str  # "call" | "ticket" | "email" | "slack" | "teams" | "crm"
    position: str | None
    phrasing: str | None
    account_id: str = ""  # the account this fact's claim is actually about
    supersedes: str | None = None
    extra_source_kind: str | None = None  # cross-source dedup: a 2nd kind
    extra_speaker: str | None = None


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-").replace(".", "")


def _build_accounts(
    rng: random.Random, num_accounts: int
) -> tuple[list[WorldAccount], dict[str, WorldAccount]]:
    stems = list(ACCOUNT_STEMS)
    rng.shuffle(stems)
    accounts: list[WorldAccount] = []
    by_id: dict[str, WorldAccount] = {}
    crm_counter = 4001
    for i in range(num_accounts):
        stem = stems[i % len(stems)]
        region = rng.choice(REGIONS)
        has_children = rng.random() < 0.4
        base_arr = rng.randint(150, 2400) * 1000
        if has_children:
            parent_name = f"{stem} {rng.choice(SYSTEM_SUFFIXES)}"
            parent_id = f"w-{_slug(stem)}"
            domain = f"{_slug(stem).replace('-', '')}.org"
            parent = WorldAccount(
                id=parent_id,
                crm_id=f"A-{crm_counter}",
                name=parent_name,
                kind="system",
                parent=None,
                domain=domain,
                arr=0,
                region=region,
                aliases=[stem],
            )
            crm_counter += 1
            accounts.append(parent)
            by_id[parent.id] = parent
            n_children = rng.choice([1, 2])
            for c in range(n_children):
                fac_name = (
                    f"{stem} {rng.choice(FACILITY_SUFFIXES)}"
                    if c == 0
                    else f"{stem} {['North', 'East', 'South', 'West'][c % 4]} Hospital"
                )
                fac_id = f"{parent_id}-{c + 1}"
                fac_arr = base_arr // (c + 2)
                fac = WorldAccount(
                    id=fac_id,
                    crm_id=f"A-{crm_counter}",
                    name=fac_name,
                    kind="facility",
                    parent=parent_id,
                    domain=domain,
                    arr=fac_arr,
                    region=region,
                    aliases=[fac_name.replace(f"{stem} ", "")],
                )
                crm_counter += 1
                accounts.append(fac)
                by_id[fac.id] = fac
        else:
            suffix = rng.choice(STANDALONE_SUFFIXES + FACILITY_SUFFIXES)
            name = f"{stem} {suffix}"
            acc_id = f"w-{_slug(stem)}"
            domain = f"{_slug(stem).replace('-', '')}.org"
            acc = WorldAccount(
                id=acc_id,
                crm_id=f"A-{crm_counter}",
                name=name,
                kind="facility" if suffix in FACILITY_SUFFIXES else "standalone",
                parent=None,
                domain=domain,
                arr=base_arr,
                region=region,
                aliases=[stem],
            )
            crm_counter += 1
            accounts.append(acc)
            by_id[acc.id] = acc
    return accounts, by_id


def _build_contacts(
    rng: random.Random, accounts: list[WorldAccount]
) -> tuple[list[WorldContact], dict[str, list[WorldContact]]]:
    contacts: list[WorldContact] = []
    by_account: dict[str, list[WorldContact]] = {}
    crm_counter = 6001
    used_names: set[str] = set()
    leaf_accounts = [a for a in accounts if a.kind != "system"]
    for acc in leaf_accounts:
        n = rng.choice([1, 1, 2])
        for _ in range(n):
            for _attempt in range(10):
                name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                if name not in used_names:
                    used_names.add(name)
                    break
            first = name.split()[0].lower()
            last = name.split()[1].lower()
            title = rng.choice(CONTACT_TITLES)
            contact = WorldContact(
                id=f"w-{_slug(name)}",
                crm_id=f"C-{crm_counter}",
                name=name,
                account=acc.id,
                title=title,
                email=f"{first[0]}{last}@{acc.domain}",
            )
            crm_counter += 1
            contacts.append(contact)
            by_account.setdefault(acc.id, []).append(contact)
    return contacts, by_account


def _build_internal_people(
    rng: random.Random, seed: int, supplier_domain: str
) -> list[WorldPerson]:
    people: list[WorldPerson] = []
    used: set[str] = set()
    for role, territory in INTERNAL_ROLE_POOL:
        for _attempt in range(10):
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name not in used:
                used.add(name)
                break
        first = name.split()[0].lower()
        last = name.split()[1].lower()
        people.append(
            WorldPerson(
                id=f"w-{_slug(name)}",
                name=name,
                role=role,
                territory=territory,
                email=f"{first[0]}.{last}@{supplier_domain}",
            )
        )
    return people


def _reps(people: list[WorldPerson]) -> list[WorldPerson]:
    return [p for p in people if p.role == "account rep"]


# --------------------------------------------------------------------------
# Fact templates -- one or more per fact type, each returning
# (claim, entities, phrasing). `phrasing` follows the hand-built corpus's
# convention: a short quoted-register note, or `None` for "state plainly".
# --------------------------------------------------------------------------


def _tpl_complaint(rng, acct, product, d):
    days = rng.choice([2, 3, 4, 5])
    claim = f"{acct.name} reported a {product['name']} delivery arriving {days} days late on {d}."
    phrasing = rng.choice(
        [f'"{days} days late"', f'"that shipment showed up {days} days behind"', None]
    )
    return claim, [acct.id, product["id"]], phrasing


def _tpl_request(rng, acct, product, d):
    feature = rng.choice(
        [
            "a night-mode alarm volume setting",
            "a Spanish-language quick-start card",
            "an EHR integration timeline",
            "additional loaner units during peak census",
            "a firmware update schedule in writing",
        ]
    )
    claim = f"{acct.name} asked for {feature} on {product['name']}."
    return claim, [acct.id, product["id"]], None


def _tpl_praise(rng, acct, product, d):
    aspect = rng.choice(
        [
            "field engineer response time",
            "the onboarding training",
            "consumable reorder turnaround",
            "the HL7 feed reliability since go-live",
        ]
    )
    claim = f"{acct.name} praised {aspect} for {product['name']} on {d}."
    phrasing = '"the fastest we\'ve had from any vendor"' if "response" in aspect else None
    return claim, [acct.id, product["id"]], phrasing


def _tpl_risk(rng, acct, product, d):
    claim = f"{acct.name} said continued issues with {product['name']} would come up at renewal."
    return claim, [acct.id, product["id"]], '"this is going to come up when we talk about renewal"'


def _tpl_competitor(rng, acct, competitor, d):
    claim = f"{acct.name} said it was evaluating {competitor['name']} pumps as of {d}."
    return (
        claim,
        [acct.id, competitor["id"]],
        f'"we\'ve had {competitor["name"].split()[0]} in to demo"',
    )


def _tpl_commitment(rng, acct, rep, d):
    claim = f"{rep.name} committed to a written follow-up summary for {acct.name} by {d}."
    return claim, [acct.id], None


def _tpl_decision(rng, acct, product, d):
    claim = f"{acct.name} decided to standardize on {product['name']} by {d}."
    return claim, [acct.id, product["id"]], None


def _tpl_contact_change(rng, acct, old_contact, new_name, d):
    claim = f"{old_contact.name} left {acct.name} in {d[:7]}; {new_name} took over."
    return (
        claim,
        [acct.id],
        f'introduces themself as "taking over from {old_contact.name.split()[0]}"',
    )


def _tpl_pricing(rng, acct, product, d):
    pct = rng.choice([8, 10, 12, 15])
    claim = f"{acct.name} asked for a {pct} percent discount on {product['name']} at renewal."
    return claim, [acct.id, product["id"]], f'"something around {pct} points off"'


def _tpl_usage(rng, acct, product, d):
    n = rng.randint(4, 48)
    claim = f"{acct.name} runs {n} {product['name']} pumps as of {d}."
    return claim, [acct.id, product["id"]], None


def _tpl_question(rng, acct, product, d):
    claim = f"{acct.name} asked whether {product['name']} supports a third-party EHR integration and did not get an answer."
    return claim, [acct.id, product["id"]], '"let me check on that"'


def _tpl_other(rng, branch_name, supplier_short, d):
    claim = (
        f"The {branch_name} distribution branch was short-staffed around {d}, "
        f"per {supplier_short}'s operations manager."
    )
    return claim, [], "internal ops note, no customer or product named"


def _weighted_fact_type(rng: random.Random) -> str:
    types = list(FACT_TYPE_WEIGHTS.keys())
    weights = list(FACT_TYPE_WEIGHTS.values())
    return rng.choices(types, weights=weights, k=1)[0]


def _make_fact(
    rng: random.Random,
    fact_type: str,
    fact_id: str,
    accounts: list[WorldAccount],
    contacts_by_account: dict[str, list[WorldContact]],
    reps: list[WorldPerson],
    branch_names: list[str],
    d: date,
    products: list[dict[str, Any]],
    competitors: list[dict[str, Any]],
    supplier_short: str,
    forced_account: WorldAccount | None = None,
) -> WorldFact | None:
    """`forced_account`, when given, pins the fact to that account instead
    of picking one at random -- used when the CALLER already knows which
    account the surrounding source (e.g. one specific CRM activity note)
    belongs to, so the fact's claim and the source it lands in are never
    about two different accounts."""
    d_iso = d.isoformat()
    eligible_accounts = [a for a in accounts if a.kind != "system"]
    acct = forced_account or rng.choice(eligible_accounts)
    contacts = contacts_by_account.get(acct.id) or []
    speaker = contacts[0].id if contacts else rng.choice(reps).id
    rep = rng.choice(reps)

    if fact_type == "complaint":
        product = products[1] if rng.random() < 0.7 else products[2]
        claim, entities, phrasing = _tpl_complaint(rng, acct, product, d_iso)
    elif fact_type == "request":
        product = rng.choice(products[1:])
        claim, entities, phrasing = _tpl_request(rng, acct, product, d_iso)
    elif fact_type == "praise":
        product = rng.choice(products[1:])
        claim, entities, phrasing = _tpl_praise(rng, acct, product, d_iso)
    elif fact_type == "risk":
        product = products[-1]
        claim, entities, phrasing = _tpl_risk(rng, acct, product, d_iso)
    elif fact_type == "competitor_mention":
        competitor = rng.choice(competitors)
        claim, entities, phrasing = _tpl_competitor(rng, acct, competitor, d_iso)
    elif fact_type == "commitment":
        claim, entities, phrasing = _tpl_commitment(rng, acct, rep, d_iso)
        speaker = rep.id
    elif fact_type == "decision":
        product = rng.choice(products[1:3])
        claim, entities, phrasing = _tpl_decision(rng, acct, product, d_iso)
    elif fact_type == "contact_change":
        if not contacts:
            return None
        old_contact = contacts[0]
        new_name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        claim, entities, phrasing = _tpl_contact_change(rng, acct, old_contact, new_name, d_iso)
        speaker = old_contact.id
    elif fact_type == "pricing":
        product = rng.choice(products[1:3])
        claim, entities, phrasing = _tpl_pricing(rng, acct, product, d_iso)
    elif fact_type == "usage":
        product = rng.choice(products[1:3])
        claim, entities, phrasing = _tpl_usage(rng, acct, product, d_iso)
    elif fact_type == "question":
        product = products[2]
        claim, entities, phrasing = _tpl_question(rng, acct, product, d_iso)
    else:  # "other" -- internal ops note, no customer/product named
        branch = rng.choice(branch_names)
        claim, entities, phrasing = _tpl_other(rng, branch, supplier_short, d_iso)
        speaker = rep.id

    return WorldFact(
        id=fact_id,
        type=fact_type,
        claim=claim,
        entities=entities,
        date=d_iso,
        speaker=speaker,
        kind="",  # filled in by the caller once the destination kind is picked
        position=rng.choice(["early", "middle", "late"]),
        phrasing=phrasing,
        account_id=acct.id,
    )


PROSE_KINDS = ["call", "ticket", "email", "slack", "teams"]


def _fact_bearing_per_kind(per_kind: int) -> int:
    return max(4, round(per_kind * 0.35))


def balanced_type_mix(per_kind: int) -> dict[str, int]:
    """Default `--type-mix balanced` profile: every production `fact_type`
    (`FACT_TYPES`, 12 entries, exactly `portolan.extract.schema.FactType`)
    gets an equal floor share of one batch's fact-bearing prose slots
    (`_fact_bearing_per_kind(per_kind) * len(PROSE_KINDS)` -- the same total
    `_plan_facts` sizes itself to by default) -- corrects the natural
    `FACT_TYPE_WEIGHTS` skew (pricing/risk/question/competitor_mention/
    contact_change are the rare ones -- see the reseeded-Halvard-corpus
    counts in `01-spec/synthetic-data-pipeline-2026-09-19.md`'s "Added
    2026-09-21" section) without inflating how many network calls a batch
    makes."""
    total_prose_facts = _fact_bearing_per_kind(per_kind) * len(PROSE_KINDS)
    share = max(1, total_prose_facts // len(FACT_TYPES))
    return dict.fromkeys(FACT_TYPES, share)


def parse_type_mix(spec: str | None, per_kind: int) -> dict[str, int] | None:
    """Parse a `--type-mix` CLI value.

    - `None`/empty -> `None` (no forcing; `_plan_facts` keeps its natural
      `FACT_TYPE_WEIGHTS` distribution -- unchanged default behavior).
    - the literal string `"balanced"` -> `balanced_type_mix(per_kind)`.
    - otherwise a comma-separated `type=count` list, e.g.
      `"pricing=40,competitor_mention=40,risk=40,question=40,contact_change=30"`
      -- per-type MINIMUM planted-fact counts for the batch.

    Raises `ValueError` (fail loudly, not silently) on an unknown fact type
    or a non-integer count.
    """
    if not spec:
        return None
    if spec.strip() == "balanced":
        return balanced_type_mix(per_kind)
    known = set(FACT_TYPES)
    result: dict[str, int] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"invalid --type-mix entry {pair!r}: expected type=count")
        if key not in known:
            raise ValueError(f"unknown fact type {key!r} in --type-mix (known: {sorted(known)})")
        try:
            result[key] = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"invalid count {value!r} for type {key!r} in --type-mix") from exc
    return result


def _plan_facts(
    rng: random.Random,
    accounts: list[WorldAccount],
    contacts_by_account: dict[str, list[WorldContact]],
    people: list[WorldPerson],
    branch_names: list[str],
    window_start: date,
    window_weeks: int,
    per_kind: int,
    products: list[dict[str, Any]],
    competitors: list[dict[str, Any]],
    supplier_short: str,
    type_mix: dict[str, int] | None = None,
) -> list[WorldFact]:
    reps = _reps(people) or people
    total_prose_facts = _fact_bearing_per_kind(per_kind) * len(PROSE_KINDS)

    # Forced type-mix (build instructions item 1, "the rare types need
    # hundreds of gold examples each"): when given, every type in
    # `type_mix` is guaranteed at least its minimum count among this
    # batch's planted facts -- `total_prose_facts` grows to fit the
    # minimums if they exceed the natural total, the rest of the slots
    # still draw from the natural `FACT_TYPE_WEIGHTS` distribution, and the
    # combined list is shuffled so a forced type isn't clustered at one
    # kind/date (position `i` below still drives `kind`/`date`
    # assignment). `type_mix` falsy (the default, `None`) skips all of
    # this and keeps the EXACT original per-index weighted-choice
    # behavior, so every existing caller is unaffected.
    forced_types: list[str] | None = None
    if type_mix:
        unknown = set(type_mix) - set(FACT_TYPES)
        if unknown:
            raise ValueError(f"unknown fact type(s) in type_mix: {sorted(unknown)}")
        minimum_total = sum(type_mix.values())
        total_prose_facts = max(total_prose_facts, minimum_total)
        forced_types = []
        for fact_type, n in type_mix.items():
            forced_types.extend([fact_type] * n)
        for _ in range(total_prose_facts - len(forced_types)):
            forced_types.append(_weighted_fact_type(rng))
        rng.shuffle(forced_types)

    facts: list[WorldFact] = []
    counter = 1001
    kind_cycle = list(PROSE_KINDS)
    for i in range(total_prose_facts):
        fact_type = forced_types[i] if forced_types is not None else _weighted_fact_type(rng)
        week = i % window_weeks
        day_offset = week * 7 + rng.randint(0, 6)
        d = window_start + timedelta(days=day_offset)
        fact_id = f"F{counter}"
        counter += 1
        fact = _make_fact(
            rng,
            fact_type,
            fact_id,
            accounts,
            contacts_by_account,
            reps,
            branch_names,
            d,
            products,
            competitors,
            supplier_short,
        )
        if fact is None:
            continue
        fact.kind = kind_cycle[i % len(kind_cycle)]
        facts.append(fact)

    # Recurring/resolving facts: ~15% of supersedable facts get a later
    # reversal (mirrors F11->F11b / F15->F15b in the hand-built corpus).
    supersedable = [f for f in facts if f.type in SUPERSEDABLE_TYPES]
    rng.shuffle(supersedable)
    n_resolutions = round(len(supersedable) * 0.15)
    for original in supersedable[:n_resolutions]:
        resolve_date = date.fromisoformat(original.date) + timedelta(days=rng.randint(10, 21))
        fact_id = f"F{counter}"
        counter += 1
        new_claim = (
            original.claim.replace("late", "on time", 1)
            if "late" in original.claim
            else original.claim + " (updated)"
        )
        resolution = WorldFact(
            id=fact_id,
            type=original.type,
            claim=f"Update: {new_claim}",
            entities=original.entities,
            date=resolve_date.isoformat(),
            speaker=original.speaker,
            kind=original.kind,
            position=rng.choice(["early", "middle", "late"]),
            phrasing=None,
            account_id=original.account_id,
            supersedes=original.id,
        )
        facts.append(resolution)

    # Cross-source dedup: ~10% of facts also land in a second kind, with a
    # different phrasing note per occurrence (kept generic — see
    # briefs.PHRASING_BY_SOURCE for why the hand-built corpus can be more
    # precise here; this generator keeps one shared phrasing note).
    eligible_for_dedup = [f for f in facts if f.supersedes is None]
    rng.shuffle(eligible_for_dedup)
    n_dedup = round(len(eligible_for_dedup) * 0.10)
    for f in eligible_for_dedup[:n_dedup]:
        other_kinds = [k for k in PROSE_KINDS if k != f.kind]
        f.extra_source_kind = rng.choice(other_kinds)
        f.extra_speaker = f.speaker

    return facts


def _iso_date(d: date) -> str:
    return d.isoformat()


@dataclass
class SourceCounters:
    call: int = 0
    ticket: int = 0
    email: int = 0
    slack: int = 0
    teams: int = 0

    def next_id(self, kind: str) -> str:
        current = getattr(self, kind)
        current += 1
        setattr(self, kind, current)
        return f"S-{kind}-{current:03d}"


def _build_sources_section(
    rng: random.Random,
    facts: list[WorldFact],
    accounts: list[WorldAccount],
    contacts_by_account: dict[str, list[WorldContact]],
    people: list[WorldPerson],
    per_kind: int,
    window_start: date,
    window_weeks: int,
) -> dict[str, Any]:
    """Builds the `sources:` block: calls/tickets/emails get the exact
    shape `briefs.build_calls`/`build_tickets`/`build_emails` already
    expect (so those functions run unchanged); `slack`/`teams` get a new
    shape this module's own `briefs.build_chat_briefs` reads."""
    reps = _reps(people) or people
    counters = SourceCounters()
    eligible_accounts = [a for a in accounts if a.kind != "system"]

    calls_items: list[dict[str, Any]] = []
    tickets_items: list[dict[str, Any]] = []
    emails_items: list[dict[str, Any]] = []
    slack_items: list[dict[str, Any]] = []
    teams_items: list[dict[str, Any]] = []

    channel_by_account = {a.id: f"acct-{_slug(a.name)}" for a in accounts}

    def _contact_or_rep(acct_id: str) -> str:
        contacts = contacts_by_account.get(acct_id) or []
        return contacts[0].id if contacts else rng.choice(reps).id

    accounts_by_id = {a.id: a for a in eligible_accounts}

    def _place_fact(f: WorldFact, kind: str, is_extra: bool) -> None:
        speaker = f.extra_speaker if is_extra else f.speaker
        # `f.account_id` (set by `_make_fact`) is the account the fact's
        # claim is actually ABOUT -- always trust it over re-deriving from
        # the speaker, which fails silently whenever the speaker is an
        # internal rep rather than a customer contact (`commitment`/
        # `other` facts) and would otherwise route the source to a random,
        # unrelated account/channel.
        acct = accounts_by_id.get(f.account_id) or rng.choice(eligible_accounts)
        if kind == "call":
            sid = counters.next_id("call")
            rep = rng.choice(reps)
            calls_items.append(
                {
                    "id": sid,
                    "date": f.date,
                    "participants": [speaker, rep.id],
                    "account": acct.id,
                    "length_tokens": 1200,
                }
            )
        elif kind == "ticket":
            sid = counters.next_id("ticket")
            tickets_items.append(
                {"id": sid, "date": f.date, "account": acct.id, "requester": speaker}
            )
        elif kind == "email":
            sid = counters.next_id("email")
            rep = rng.choice(reps)
            emails_items.append({"id": sid, "date": f.date, "from": speaker, "to": [rep.id]})
        elif kind == "slack":
            sid = counters.next_id("slack")
            is_customer = speaker in {c.id for cs in contacts_by_account.values() for c in cs}
            slack_items.append(
                {
                    "id": sid,
                    "date": f.date,
                    "channel": channel_by_account[acct.id],
                    "account": acct.id,
                    "speaker": speaker,
                    "is_customer": is_customer,
                    "thread": rng.random() < 0.3,
                }
            )
        elif kind == "teams":
            sid = counters.next_id("teams")
            is_customer = speaker in {c.id for cs in contacts_by_account.values() for c in cs}
            teams_items.append(
                {
                    "id": sid,
                    "date": f.date,
                    "channel": channel_by_account[acct.id],
                    "account": acct.id,
                    "speaker": speaker,
                    "is_customer": is_customer,
                    "thread": rng.random() < 0.3,
                }
            )
        f_sources = f.__dict__.setdefault("_source_ids", [])
        f_sources.append((sid, kind))

    for f in facts:
        _place_fact(f, f.kind, is_extra=False)
        if f.extra_source_kind:
            _place_fact(f, f.extra_source_kind, is_extra=True)

    # `calls`/`tickets`/`emails` get their noise generated later, by
    # `briefs.build_calls`/`build_tickets`/`build_emails` themselves (the
    # `noise.count` field below is exactly what those functions already
    # read). `slack`/`teams` have no such auto-noise step in briefs.py
    # (`build_chat_briefs`, added alongside this module, just iterates
    # `items` and infers `is_noise` from "no facts") -- so every noise item
    # for those two kinds is generated here, directly into `items`, topping
    # each channel-file up toward `per_kind` total sources of that kind.
    def _add_chat_noise(kind: str, items: list[dict[str, Any]]) -> None:
        target = max(0, per_kind - len(items))
        for _ in range(target):
            acct = rng.choice(eligible_accounts)
            speaker = _contact_or_rep(acct.id)
            is_customer = speaker in {c.id for cs in contacts_by_account.values() for c in cs}
            day_offset = rng.randint(0, window_weeks * 7 - 1)
            d = window_start + timedelta(days=day_offset)
            sid = counters.next_id(kind)
            items.append(
                {
                    "id": sid,
                    "date": d.isoformat(),
                    "channel": channel_by_account[acct.id],
                    "account": acct.id,
                    "speaker": speaker,
                    "is_customer": is_customer,
                    "thread": False,
                }
            )

    _add_chat_noise("slack", slack_items)
    _add_chat_noise("teams", teams_items)

    return (
        {
            "calls": {
                "items": calls_items,
                "noise": {"count": max(0, per_kind - len(calls_items))},
            },
            "tickets": {
                "items": tickets_items,
                "noise": {"count": max(0, per_kind - len(tickets_items))},
            },
            "emails": {
                "items": emails_items,
                "noise": {"count": max(0, per_kind - len(emails_items))},
            },
            "slack": {"items": slack_items},
            "teams": {"items": teams_items},
        },
        slack_items,
        teams_items,
        channel_by_account,
    )


def fact_bearing_count(items: list[dict[str, Any]]) -> int:
    return len(items)


def _fact_sources_field(f: WorldFact) -> list[dict[str, Any]]:
    """Builds this fact's `sources:` list (the field `briefs._facts_for_source`
    scans) from the `(source_id, kind)` pairs `_build_sources_section`
    recorded on it."""
    out = []
    for sid, kind in f.__dict__.get("_source_ids", []):
        speaker = f.extra_speaker if (kind == f.extra_source_kind) else f.speaker
        entry: dict[str, Any] = {"id": sid, "type": kind, "speaker": speaker}
        if kind in ("call",):
            entry["position"] = f.position
        out.append(entry)
    return out


def generate_world(
    seed: int,
    num_accounts: int = 5,
    per_kind: int = 60,
    type_mix: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Generate one fresh, fully-synthetic truth table. Deterministic in
    `(seed, num_accounts, per_kind, type_mix)` alone.

    `type_mix`, when given, forces per-type MINIMUM planted-fact counts
    (see `parse_type_mix`/`balanced_type_mix`) instead of the natural
    `FACT_TYPE_WEIGHTS` distribution -- `None` (the default) is fully
    backward compatible."""
    rng = random.Random(seed)
    supplier, products, competitors = _pick_vendor_mythology(rng)
    accounts, accounts_by_id = _build_accounts(rng, num_accounts)
    contacts, contacts_by_account = _build_contacts(rng, accounts)
    people = _build_internal_people(rng, seed, supplier["domain"])
    branch_names = rng.sample(BRANCH_NAMES, k=min(3, len(BRANCH_NAMES)))

    window_start = date(2026, 1, 5) + timedelta(weeks=(seed % 6))
    window_weeks = 8
    window_end = window_start + timedelta(weeks=window_weeks) - timedelta(days=1)

    facts = _plan_facts(
        rng,
        accounts,
        contacts_by_account,
        people,
        branch_names,
        window_start,
        window_weeks,
        per_kind,
        products,
        competitors,
        supplier["short"],
        type_mix=type_mix,
    )
    sources, slack_items, teams_items, channel_by_account = _build_sources_section(
        rng, facts, accounts, contacts_by_account, people, per_kind, window_start, window_weeks
    )

    # ---- crm: structured accounts/opportunities/cases + prose "notes" ----
    crm_items: list[dict[str, Any]] = []
    crm_records: dict[str, list[dict[str, Any]]] = {"opportunities": [], "cases": []}
    crm_notes_items: list[dict[str, Any]] = []

    eligible_accounts = [a for a in accounts if a.kind != "system"]
    n_opportunities = max(1, round(per_kind * 0.30))
    n_cases = max(1, round(per_kind * 0.25))
    n_notes = max(1, round(per_kind * 0.25))

    opp_counter = 5001
    for i in range(n_opportunities):
        acct = eligible_accounts[i % len(eligible_accounts)]
        opp_id = f"O-{opp_counter}"
        opp_counter += 1
        stage = rng.choice(["qualification", "negotiation", "closed_won", "closed_lost"])
        close_date = (window_end + timedelta(days=rng.randint(30, 400))).isoformat()
        crm_records["opportunities"].append(
            {
                "opportunity_id": opp_id,
                "account_crm_id": acct.crm_id,
                "name": f"{acct.name} — {supplier['short']}Care renewal",
                "stage": stage,
                "close_date": close_date,
                "amount": acct.arr or rng.randint(100, 900) * 1000,
                "export_date": window_end.isoformat(),
            }
        )
        crm_items.append(
            {
                "id": f"S-crm-opp-{opp_id.replace('-', '')}",
                "object": "opportunity",
                "export_date": window_end.isoformat(),
                "crm_id": opp_id,
                "path": "crm/opportunities.csv",
            }
        )

    case_counter = 7001
    for i in range(n_cases):
        acct = eligible_accounts[i % len(eligible_accounts)]
        contacts_here = contacts_by_account.get(acct.id) or []
        contact = contacts_here[0] if contacts_here else None
        case_id = f"CASE-{case_counter}"
        case_counter += 1
        created = window_start + timedelta(days=rng.randint(0, window_weeks * 7 - 1))
        crm_records["cases"].append(
            {
                "case_id": case_id,
                "account_crm_id": acct.crm_id,
                "contact_crm_id": contact.crm_id if contact else "",
                "subject": rng.choice(
                    [
                        "Delivery timing question",
                        "Consumable reorder minimums",
                        "Portal access issue",
                        "Training request",
                        "Invoice discrepancy",
                    ]
                ),
                "status": rng.choice(["open", "closed", "escalated"]),
                "priority": rng.choice(["low", "medium", "high"]),
                "created_date": created.isoformat(),
                "closed_date": (created + timedelta(days=rng.randint(1, 14))).isoformat()
                if rng.random() < 0.6
                else "",
            }
        )
        crm_items.append(
            {
                "id": f"S-crm-case-{case_id.replace('-', '')}",
                "object": "case",
                "export_date": window_end.isoformat(),
                "crm_id": case_id,
                "path": "crm/cases.csv",
            }
        )

    # A handful of CRM-only facts (ARR/parent-chain style, no prose needed —
    # mirrors F35/F36 in the hand-built corpus) land directly on account rows.
    crm_direct_facts: list[WorldFact] = []
    counter = 9001
    for acct in eligible_accounts[: max(1, round(per_kind * 0.1))]:
        fact_id = f"F{counter}"
        counter += 1
        claim = (
            f"{acct.name}'s annual recurring revenue with {supplier['short']} is "
            f"{acct.arr} dollars as of {window_end.isoformat()}."
        )
        wf = WorldFact(
            id=fact_id,
            type="other",
            claim=claim,
            entities=[acct.id],
            date=window_end.isoformat(),
            speaker="",
            kind="crm",
            position=None,
            phrasing=None,
        )
        sid = f"S-crm-{acct.crm_id.replace('-', '')}"
        wf.__dict__["_source_ids"] = [(sid, "crm")]
        crm_direct_facts.append(wf)
        crm_items.append(
            {
                "id": sid,
                "object": "account",
                "crm_id": acct.crm_id,
                "path": "crm/accounts.csv",
            }
        )

    # CRM activity notes: prose, may carry a fact.
    note_counter = 8001
    notes_facts: list[WorldFact] = []
    n_notes_with_facts = max(1, round(n_notes * 0.4))
    for i in range(n_notes):
        acct = eligible_accounts[i % len(eligible_accounts)]
        contacts_here = contacts_by_account.get(acct.id) or []
        contact = contacts_here[0] if contacts_here else None
        author = rng.choice(_reps(people) or people)
        note_id = f"N-{note_counter}"
        note_counter += 1
        created = window_start + timedelta(days=rng.randint(0, window_weeks * 7 - 1))
        item: dict[str, Any] = {
            "id": f"S-crm-note-{note_id}",
            "date": created.isoformat(),
            "account": acct.id,
            "contact": contact.id if contact else None,
            "author": author.id,
            "note_id": note_id,
        }
        if i < n_notes_with_facts:
            fact_type = rng.choice(["complaint", "request", "pricing", "praise", "usage"])
            fid = f"F{counter}"
            counter += 1
            wf = _make_fact(
                rng,
                fact_type,
                fid,
                accounts,
                contacts_by_account,
                _reps(people) or people,
                branch_names,
                created,
                products,
                competitors,
                supplier["short"],
                forced_account=acct,
            )
            if wf is not None:
                wf.kind = "crm"
                wf.speaker = author.id
                wf.__dict__["_source_ids"] = [(item["id"], "crm")]
                notes_facts.append(wf)
        crm_notes_items.append(item)

    all_facts = facts + crm_direct_facts + notes_facts

    truth: dict[str, Any] = {
        "meta": {
            "version": "synth-0.1",
            "schema_version": "0.1",
            "seed": seed,
            "window": {"start": window_start.isoformat(), "end": window_end.isoformat()},
            "supplier": supplier,
            "products": products,
            "competitors": competitors,
            "internal_people": [
                {
                    "id": p.id,
                    "name": p.name,
                    "role": p.role,
                    "territory": p.territory,
                    "email": p.email,
                }
                for p in people
            ],
            # Each branch serves a round-robin slice of the eligible
            # accounts (index i, i+len(branch_names), ...) -- decorative
            # metadata only (no code path reads `serves` back; the "other"
            # fact template picks a branch name directly from
            # `branch_names`), but real distinct accounts per branch
            # rather than "first branch gets everyone, the rest get
            # nobody" (the bug this replaced).
            "branches": [
                {
                    "id": _slug(name),
                    "name": f"{name} distribution branch",
                    "serves": [
                        a.id for j, a in enumerate(eligible_accounts) if j % len(branch_names) == i
                    ],
                }
                for i, name in enumerate(branch_names)
            ],
        },
        "accounts": [
            {
                "id": a.id,
                "crm_id": a.crm_id,
                "name": a.name,
                "kind": a.kind,
                "parent": a.parent,
                "domain": a.domain,
                "arr": a.arr,
                "region": a.region,
                "aliases": a.aliases,
            }
            for a in accounts
        ],
        "contacts": [
            {
                "id": c.id,
                "crm_id": c.crm_id,
                "name": c.name,
                "account": c.account,
                "title": c.title,
                "email": c.email,
            }
            for c in contacts
        ],
        "facts": [
            {
                "id": f.id,
                "type": f.type,
                "claim": f.claim,
                "entities": f.entities,
                "date": f.date,
                "sources": _fact_sources_field(f),
                **({"phrasing": f.phrasing} if f.phrasing else {}),
                **({"supersedes": f.supersedes} if f.supersedes else {}),
            }
            for f in all_facts
        ],
        "sources": {
            **sources,
            "crm": {"items": crm_items},
            "crm_notes": {"items": crm_notes_items},
        },
        "crm_records": crm_records,
    }
    return truth
