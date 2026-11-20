"""Composed name pools for the v2 world simulator, split-disjoint by hash.

`01-spec/world-generator-v2-2026-09-22.md` section 6: "Composed first-name
and last-name pools (at least 400 x 600, several cultures), account-name
stems (v1's 812, widened), product and competitor name generators. The
pools are split by hash into train, validation and test, so no name
crosses splits."

**Composed, not hand-typed.** Like v1's `ACCOUNT_STEMS` (`corpus/generator/
worlds.py`: a fixed prefix list x a fixed suffix list, filtered against a
hand-corpus blocklist), every pool below is a fixed onset-fragment list x a
fixed coda-fragment list per culture family, combined by `_compose()`. This
is a synthetic-data engineering technique (also used by procedural name
generators generally), not a claim of ethnographic accuracy -- the culture
labels are phonetic-style tags, chosen for diversity across the training
corpus (spec: "several cultures"), not a representation of any real
population's naming conventions.

**Split partitioning.** `pool_for_split()` buckets every name in a pool
into train/validation/test by hashing `f"{salt}:{name}"` (sha256, taken
mod 100) -- deterministic, independent of any world's `seed`, and stable
across process runs. Because `simulate_world()` only ever draws a first
name from `pool_for_split(FIRST_NAMES, split, "first_name")`, a last name
from `pool_for_split(LAST_NAMES, split, "last_name")`, an account stem
from `pool_for_split(ACCOUNT_STEMS, split, "account_stem")`, and so on,
the full assembled names (person = first + last, account = stem + suffix,
vendor = prefix + suffix, competitor = prefix + suffix, product = line +
tier) can never collide across splits: the component pools themselves
never overlap. Vendors and competitors are partitioned on the *prefix*
(`vendor_prefixes_for_split`), because the prefix is the short name,
alias and email domain that documents repeat (WP3 review). This is enforced by
construction, not just checked after the fact -- `test_world_generator_v2.
py::test_name_pools_disjoint_across_splits` checks it holds for two
different seeds per split (the partition doesn't depend on seed at all,
so this is really checking "the same split always gets the same names,
and no other split's names").
"""

from __future__ import annotations

import hashlib
import itertools
from enum import StrEnum
from pathlib import Path


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# Round-1 split ratios roughly match the spec's 12 train : 2 validation :
# 3 test world mix (section 6's table) -- not exact (that's a world count,
# this is a name-pool bucket), just in the same neighborhood so no split
# is starved of names.
_SPLIT_BOUNDARIES: list[tuple[int, Split]] = [
    (70, Split.TRAIN),
    (85, Split.VALIDATION),
    (100, Split.TEST),
]


def split_bucket(name: str, salt: str) -> Split:
    """Deterministic, seed-independent split assignment for one name
    string. Same `(name, salt)` always returns the same split, on any
    machine, in any process."""
    digest = hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()
    bucket = int(digest, 16) % 100
    for ceiling, split in _SPLIT_BOUNDARIES:
        if bucket < ceiling:
            return split
    return Split.TEST  # pragma: no cover -- unreachable, ceiling always 100


def pool_for_split(pool: list[str], split: Split, salt: str) -> list[str]:
    """The subset of `pool` assigned to `split` under `salt`. Two
    different `salt` values partition the same `pool` independently
    (e.g. `FIRST_NAMES` uses salt `"first_name"`, `LAST_NAMES` uses
    `"last_name"`) -- that's fine, since what must never collide across
    splits is the *fully assembled* name, and each component pool is
    partitioned on its own values."""
    return [name for name in pool if split_bucket(name, salt) == split]


def _compose(onsets: list[str], codas: list[str], target: int) -> list[str]:
    """`onsets x codas`, in `itertools.product` order, deduplicated
    case-insensitively, capped at `target` entries. Deterministic: no
    randomness, so the pool is a fixed module-level constant."""
    names: list[str] = []
    seen: set[str] = set()
    for onset, coda in itertools.product(onsets, codas):
        name = onset + coda
        key = name.lower()
        if key in seen or len(name) < 3:
            continue
        seen.add(key)
        names.append(name)
        if len(names) >= target:
            break
    return names


# --------------------------------------------------------------------------
# First names -- combinatorial per culture family, onset x coda.
# --------------------------------------------------------------------------

_FIRST_NAME_FRAGMENTS: dict[str, tuple[list[str], list[str]]] = {
    "anglo": (
        [
            "Al",
            "Ben",
            "Cal",
            "Dan",
            "Ed",
            "Fran",
            "Gil",
            "Har",
            "Ian",
            "Jer",
            "Kel",
            "Len",
            "Mar",
            "Nat",
            "Owe",
            "Pat",
            "Quin",
            "Ros",
            "Sam",
            "Tom",
            "Vic",
            "Wal",
        ],
        [
            "an",
            "ard",
            "by",
            "den",
            "ett",
            "ford",
            "iel",
            "in",
            "land",
            "ley",
            "lyn",
            "man",
            "on",
            "rick",
            "son",
            "ton",
            "well",
            "wick",
            "wyn",
        ],
    ),  # fmt: skip
    "hispanic": (
        [
            "Al",
            "Ber",
            "Car",
            "Dol",
            "Est",
            "Fer",
            "Gua",
            "Ig",
            "Jav",
            "Lor",
            "Mar",
            "Nor",
            "Oct",
            "Pat",
            "Ra",
            "Sal",
            "Ter",
            "Val",
        ],
        [
            "ando",
            "berto",
            "cia",
            "dro",
            "elia",
            "ez",
            "ia",
            "iel",
            "ina",
            "io",
            "isa",
            "ito",
            "ela",
            "onso",
            "osa",
            "udio",
        ],
    ),  # fmt: skip
    "east_asian": (
        [
            "Bo",
            "Chen",
            "Feng",
            "Hui",
            "Jin",
            "Lei",
            "Mei",
            "Ning",
            "Qing",
            "Ru",
            "Shu",
            "Tao",
            "Wei",
            "Xin",
            "Yan",
            "Zhi",
            "Min",
            "Hao",
        ],
        ["an", "hua", "jun", "lin", "mei", "ming", "ping", "qi", "wen", "xia", "ying", "yu"],
    ),  # fmt: skip
    "south_asian": (
        [
            "Am",
            "Ani",
            "Chan",
            "Dee",
            "Gau",
            "Ish",
            "Jas",
            "Kav",
            "Lak",
            "Man",
            "Nee",
            "Pri",
            "Raj",
            "Sun",
            "Tar",
            "Vik",
            "Ash",
            "Rav",
        ],
        [
            "an",
            "esh",
            "ika",
            "ish",
            "ita",
            "iya",
            "jit",
            "meet",
            "nav",
            "pal",
            "ram",
            "ta",
            "tha",
            "veer",
            "ya",
        ],
    ),  # fmt: skip
    "arabic": (
        ["Am", "Fa", "Ha", "Ka", "La", "Ma", "Na", "Om", "Ra", "Sa", "Ta", "Ya", "Za", "Za", "Ha"],
        [
            "dia",
            "eed",
            "eem",
            "ir",
            "isa",
            "lah",
            "man",
            "mid",
            "reen",
            "sim",
            "wan",
            "ya",
            "zan",
            "sen",
        ],
    ),  # fmt: skip
    "west_african": (
        [
            "Ade",
            "Chi",
            "Efe",
            "Fol",
            "Ibi",
            "Ken",
            "Ngo",
            "Obi",
            "Sef",
            "Tay",
            "Uzo",
            "Yem",
            "Emeka",
            "Zola",
        ],
        [
            "chukwu",
            "dayo",
            "kemi",
            "lola",
            "nna",
            "nso",
            "oma",
            "seun",
            "tunde",
            "wale",
            "goziem",
            "ada",
        ],
    ),  # fmt: skip
    "slavic": (
        [
            "An",
            "Bog",
            "Dar",
            "El",
            "Ig",
            "Kat",
            "Lud",
            "Mir",
            "Nat",
            "Ol",
            "Rad",
            "Sve",
            "Vla",
            "Yur",
        ],
        ["ana", "dan", "eta", "imir", "ina", "islav", "ka", "mil", "nya", "slav", "usha", "yana"],
    ),  # fmt: skip
}

_FIRST_NAMES_PER_CULTURE = 65


def _dedupe_case_insensitive(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


FIRST_NAMES: list[str] = []
for _culture, (_onsets, _codas) in _FIRST_NAME_FRAGMENTS.items():
    FIRST_NAMES.extend(_compose(_onsets, _codas, _FIRST_NAMES_PER_CULTURE))
del _culture, _onsets, _codas
FIRST_NAMES = _dedupe_case_insensitive(FIRST_NAMES)

# --------------------------------------------------------------------------
# Last names -- combinatorial where the culture's real naming convention is
# itself compound (anglo place-name-style, hispanic, slavic root+suffix,
# west african), literal common surnames where compounding would misfire
# (east asian, south asian, arabic).
# --------------------------------------------------------------------------

_LAST_NAME_FRAGMENTS: dict[str, tuple[list[str], list[str]]] = {
    "anglo": (
        [
            "Ash",
            "Black",
            "Brad",
            "Clay",
            "Dun",
            "East",
            "Fair",
            "Gold",
            "Hart",
            "Iron",
            "Kirk",
            "Lang",
            "Moor",
            "North",
            "Oak",
            "Penn",
            "Rye",
            "Stone",
            "Thorn",
            "West",
            "Wood",
            "York",
        ],
        [
            "bourne",
            "brook",
            "burg",
            "by",
            "don",
            "field",
            "ford",
            "gate",
            "grove",
            "hall",
            "land",
            "ley",
            "more",
            "ridge",
            "shaw",
            "ton",
            "way",
            "well",
            "wick",
            "worth",
        ],
    ),  # fmt: skip
    "hispanic": (
        [
            "Agu",
            "Cas",
            "Del",
            "Esp",
            "Fer",
            "Gar",
            "Guz",
            "Her",
            "Igl",
            "Jim",
            "Lop",
            "Mar",
            "Nav",
            "Oro",
            "Per",
            "Quin",
            "Riv",
            "San",
            "Tor",
            "Val",
        ],
        [
            "ado",
            "ales",
            "ana",
            "ares",
            "eda",
            "era",
            "erra",
            "ez",
            "illo",
            "inez",
            "ita",
            "onda",
            "oza",
            "uela",
            "ueras",
            "undo",
        ],
    ),  # fmt: skip
    "slavic": (
        [
            "Volk",
            "Ivan",
            "Petr",
            "Sokol",
            "Med",
            "Gorb",
            "Kuz",
            "Bel",
            "Chern",
            "Zolot",
            "Nikol",
            "Rom",
            "Bar",
            "Grig",
        ],
        ["ov", "ova", "enko", "sky", "ski", "chuk", "vich", "in", "yev", "yeva", "ovich"],
    ),  # fmt: skip
    "west_african": (
        ["Ade", "Oke", "Olu", "Chi", "Eze", "Oko", "Uzo", "Nna", "Obi", "Chu", "Bal", "Ade"],
        [
            "wale",
            "yemi",
            "tunde",
            "chukwu",
            "nna",
            "goziem",
            "kanma",
            "ndu",
            "ezinne",
            "onwu",
            "ogun",
            "afor",
        ],
    ),  # fmt: skip
}

_LAST_NAMES_PER_COMPOUND_CULTURE = 155

LAST_NAMES: list[str] = []
for _culture, (_onsets, _codas) in _LAST_NAME_FRAGMENTS.items():
    LAST_NAMES.extend(_compose(_onsets, _codas, _LAST_NAMES_PER_COMPOUND_CULTURE))
del _culture, _onsets, _codas

# Literal common surnames -- east asian, south asian, arabic. Short,
# well-known forms; compounding these onset x coda would produce
# unrecognizable strings rather than plausible ones.
_LAST_NAMES_LITERAL: dict[str, list[str]] = {
    "east_asian": [
        "Chen",
        "Wang",
        "Li",
        "Zhang",
        "Liu",
        "Yang",
        "Huang",
        "Zhao",
        "Wu",
        "Zhou",
        "Xu",
        "Sun",
        "Ma",
        "Zhu",
        "Hu",
        "Guo",
        "He",
        "Gao",
        "Lin",
        "Luo",
        "Zheng",
        "Liang",
        "Song",
        "Xie",
        "Tang",
        "Han",
        "Cao",
        "Deng",
        "Feng",
        "Peng",
        "Zeng",
        "Xiao",
        "Tian",
        "Dong",
        "Pan",
        "Yuan",
        "Cai",
        "Jiang",
        "Yu",
        "Du",
        "Ye",
        "Cheng",
        "Wei",
        "Su",
        "Lu",
        "Ding",
        "Ren",
        "Shen",
        "Yao",
        "Fan",
        "Gu",
        "Kong",
        "Chai",
        "Bai",
        "Wan",
        "Kim",
        "Lee",
        "Park",
        "Choi",
        "Jung",
        "Kang",
        "Cho",
        "Yoon",
        "Jang",
        "Lim",
        "Han",
        "Oh",
        "Seo",
        "Shin",
        "Kwon",
        "Hwang",
        "Nguyen",
        "Tran",
        "Le",
        "Pham",
        "Hoang",
        "Phan",
        "Vu",
        "Dang",
        "Bui",
        "Do",
        "Ho",
        "Ngo",
        "Duong",
        "Ly",
        "Vo",
        "Trinh",
        "Dinh",
        "Mai",
        "Ta",
    ],
    "south_asian": [
        "Sharma",
        "Patel",
        "Gupta",
        "Reddy",
        "Rao",
        "Nair",
        "Iyer",
        "Singh",
        "Kumar",
        "Verma",
        "Mehta",
        "Shah",
        "Chopra",
        "Malhotra",
        "Bose",
        "Chatterjee",
        "Banerjee",
        "Mukherjee",
        "Desai",
        "Joshi",
        "Pillai",
        "Menon",
        "Naidu",
        "Rana",
        "Thakur",
        "Yadav",
        "Chauhan",
        "Bhatt",
        "Agarwal",
        "Kapoor",
        "Khanna",
        "Saxena",
        "Trivedi",
        "Pandey",
        "Mishra",
        "Dutta",
        "Sen",
        "Ghosh",
        "Das",
        "Roy",
        "Kaur",
        "Gill",
        "Bhatia",
        "Rathore",
        "Choudhary",
        "Krishnan",
        "Subramaniam",
        "Rai",
        "Dev",
    ],
    "arabic": [
        "Hassan",
        "Khalil",
        "Haddad",
        "Nasser",
        "Saleh",
        "Youssef",
        "Karam",
        "Aziz",
        "Rashid",
        "Mansour",
        "Farouk",
        "Habib",
        "Kassem",
        "Sabbagh",
        "Najjar",
        "Hakim",
        "Zaman",
        "Saad",
        "Fares",
        "Jaber",
        "Marwan",
        "Qassem",
        "Tahan",
        "Shami",
        "Aboud",
        "Ammar",
        "Darwish",
        "Ghannam",
        "Halabi",
        "Idris",
        "Jarrah",
        "Kanaan",
        "Lahham",
        "Matar",
        "Nassar",
        "Odeh",
        "Qadir",
        "Rahal",
        "Salameh",
        "Tannous",
        "Wahba",
        "Zayat",
        "Al-Amin",
        "Al-Faruqi",
        "Al-Hadid",
        "Al-Karim",
        "Al-Sayed",
        "Al-Tayeb",
        "Al-Rashid",
        "Al-Nasser",
    ],
}

for _names in _LAST_NAMES_LITERAL.values():
    LAST_NAMES.extend(_names)
del _names
LAST_NAMES = _dedupe_case_insensitive(LAST_NAMES)

# --------------------------------------------------------------------------
# Exam and prompt blocklist (WP3 review, 2026-09-22). Spec section 4,
# "Names": "Blocklists: every name in `corpus/truth-table.yaml`". v1's
# `_HAND_CORPUS_STEMS` covers the Halvard exam's account stems; the supplier,
# product and competitor names are added here, plus the invented account
# names used as worked examples in the production extraction prompt
# (`portolan/extract/schema.py` `TYPE_PRECEDENCE_RULES_V6`, `FACT_TYPES`) --
# a training world whose vendor is "Fernbridge" or whose contact is "X
# Ashgrove" would put exam and prompt names into training documents. Every
# pool below is filtered through `_allowed()`, and so is the composed
# last-name pool above; `tests/test_world_generator_v2_wp3.py` re-derives
# the exam's name tokens from the truth table itself and checks no pool
# entry matches one. Entries are compared lowercased with digits removed,
# so a near-copy such as "Corvane2" is caught too.
# --------------------------------------------------------------------------

from corpus.generator.worlds import _HAND_CORPUS_STEMS  # noqa: E402

EXAM_AND_PROMPT_BLOCKLIST: frozenset[str] = frozenset(
    set(_HAND_CORPUS_STEMS)
    | {
        # Halvard exam supplier, products, competitors (truth-table meta).
        "halvard",
        "halvardcare",
        "tidewater",
        "corvane",
        "brightline",
        # Production prompt worked-example names.
        "fernbridge",
        "oakhollow",
        "cedarvale",
        "springfield",
    }
)


def _normalize_for_blocklist(name: str) -> str:
    return "".join(ch for ch in name.lower() if not ch.isdigit()).strip()


def _allowed(name: str) -> bool:
    """False for a blocklisted name or any name carrying a digit (a digit
    in a person, account, vendor, product-line or competitor name reads
    as a generator artifact -- defect 3 of the WP3 check note)."""
    if any(ch.isdigit() for ch in name):
        return False
    return _normalize_for_blocklist(name) not in EXAM_AND_PROMPT_BLOCKLIST


FIRST_NAMES = [n for n in FIRST_NAMES if _allowed(n)]
LAST_NAMES = [n for n in LAST_NAMES if _allowed(n)]

# --------------------------------------------------------------------------
# Account-name stems -- v1's 812 (`corpus.generator.worlds.ACCOUNT_STEMS`),
# widened. Reused, not modified, per this project's build rule ("Do not
# touch v1 files except to import shared helpers"). Widened with a second
# combinatorial layer over a disjoint prefix/suffix set, still filtered
# against v1's own hand-corpus blocklist.
# --------------------------------------------------------------------------

from corpus.generator.worlds import ACCOUNT_STEMS as _V1_ACCOUNT_STEMS  # noqa: E402

# "Pinecrest2" removed in the WP3 review: it put a digit into 15 account
# stems ("Pinecrest2brook") and is a near-copy of a Halvard exam stem.
_V2_STEM_PREFIXES = [
    "Bay", "Cedar", "Delta", "Evergreen", "Farview", "Glacier", "Hemlock",
    "Ivy", "Juniper", "Knoll", "Ledge", "Marsh", "Nightingale", "Overlook",
    "Quarry", "Rowan", "Spruce", "Timber", "Union", "Valley", "Whisper",
]  # fmt: skip
_V2_STEM_SUFFIXES = [
    "brook", "crest", "dale", "fall", "glade", "harbor", "isle", "knoll",
    "meadow", "point", "reach", "shore", "spring", "vale", "watch",
]  # fmt: skip

ACCOUNT_STEMS: list[str] = [s for s in _V1_ACCOUNT_STEMS if _allowed(s)]
_seen_stems = {s.lower() for s in ACCOUNT_STEMS}
for _prefix in _V2_STEM_PREFIXES:
    for _suffix in _V2_STEM_SUFFIXES:
        _combo = f"{_prefix}{_suffix}"
        if not _allowed(_combo) or _combo.lower() in _seen_stems:
            continue
        _seen_stems.add(_combo.lower())
        ACCOUNT_STEMS.append(_combo)
del _prefix, _suffix, _combo, _seen_stems

# Facility-name qualifiers (WP3 review, defect 3). A health system's
# facilities used to be named "<stem> <suffix> <n>" ("Marrowbrook General
# Hospital 1"), and the digit reached the contacts' email domains. Children
# of one system now draw distinct (qualifier, suffix) pairs from
# `FACILITY_QUALIFIERS x pack.account_suffixes_facility` without
# replacement -- the combinatorial pool the Halvard exam's own naming uses
# ("Larkspur North Hospital", "Ashgrove Riverside Hospital"). These are
# generic place words shared by every split, like the pack's suffixes; the
# split-partitioned stem is what keeps an assembled name inside one split.
FACILITY_QUALIFIERS: list[str] = [
    "North", "South", "East", "West", "Central", "Riverside", "Lakeside",
    "Valley", "Hillside", "Parkview", "Harborview", "Memorial",
]  # fmt: skip

# --------------------------------------------------------------------------
# Vendor / product-line / competitor name generators -- disjoint from v1's
# fixed banks (`VENDOR_BANK`, `PRODUCT_LINE_NAMES`, `COMPETITOR_BANK` in
# `corpus.generator.worlds`), widened combinatorially the same way.
# --------------------------------------------------------------------------

#
# **Split by prefix, not by full name** (WP3 review). The prefix is what a
# document actually repeats: the vendor's short name and its email domain
# (`simulator._build_vendor` uses `vendor_name.split()[0]`) and a
# competitor's alias. Partitioning the full composed names let
# "Meridian Medical" land in train and "Meridian Clinical" in test, so 19 of
# 20 vendor prefixes and 7 of 12 competitor prefixes (and their domains)
# crossed splits. `vendor_prefixes_for_split` / `competitor_prefixes_for_
# split` partition the prefix itself. The hand lists are widened
# combinatorially from fragment lists this module already has (no new
# vocabulary), then filtered against the blocklist and against every other
# name pool, so a vendor can never share a string with an account stem or a
# person's surname.

_VENDOR_PREFIXES_HAND = [
    "Meridian", "Solace", "Northlight", "Anchorpoint", "Cascadia", "Vantage",
    "Ashworth", "Brightwell", "Clearwater", "Dovetail", "Everline",
    "Farhold", "Greystone", "Harborlight", "Ironbridge", "Lanternhouse",
    "Moorland",
]  # fmt: skip
_VENDOR_SUFFIXES = [
    "Medical", "Health Devices", "Biomedical", "Infusion", "Health Systems",
    "Clinical", "Healthtech", "MedTech",
]  # fmt: skip

_COMPETITOR_PREFIXES_HAND = [
    "Westhaven", "Palisade", "Orchard", "Thornfield", "Grayling", "Ambervale",
    "Silverline", "Haldane", "Brightfield", "Winterhold", "Larkbrook",
]  # fmt: skip
_COMPETITOR_SUFFIXES = ["Devices", "Medical", "Health Systems", "Biomedical"]


def _combinatorial_prefixes(
    hand: list[str], onsets: list[str], codas: list[str], exclude: set[str]
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set(exclude)
    composed = [o + c for o, c in itertools.product(onsets, codas) if o[-1].lower() != c[0].lower()]
    for candidate in [*hand, *composed]:
        key = candidate.lower()
        if key in seen or not _allowed(candidate):
            continue
        seen.add(key)
        out.append(candidate)
    return out


_OTHER_NAME_KEYS = {n.lower() for n in [*FIRST_NAMES, *LAST_NAMES, *ACCOUNT_STEMS]}

VENDOR_PREFIXES: list[str] = _combinatorial_prefixes(
    _VENDOR_PREFIXES_HAND,
    _LAST_NAME_FRAGMENTS["anglo"][0],
    _V2_STEM_SUFFIXES,
    exclude=_OTHER_NAME_KEYS,
)
COMPETITOR_PREFIXES: list[str] = _combinatorial_prefixes(
    _COMPETITOR_PREFIXES_HAND,
    _V2_STEM_PREFIXES,
    _LAST_NAME_FRAGMENTS["anglo"][1],
    exclude=_OTHER_NAME_KEYS | {p.lower() for p in VENDOR_PREFIXES},
)


def vendor_prefixes_for_split(split: Split) -> list[str]:
    return pool_for_split(VENDOR_PREFIXES, split, "vendor_prefix")


def competitor_prefixes_for_split(split: Split) -> list[str]:
    return pool_for_split(COMPETITOR_PREFIXES, split, "competitor_prefix")


VENDOR_SUFFIXES: list[str] = list(_VENDOR_SUFFIXES)
COMPETITOR_SUFFIXES: list[str] = list(_COMPETITOR_SUFFIXES)

# Full composed lists, kept for callers that want every name; draw through
# the `*_prefixes_for_split` functions to stay split-disjoint on the prefix.
VENDOR_NAMES: list[str] = [f"{p} {s}" for p in VENDOR_PREFIXES for s in VENDOR_SUFFIXES]
COMPETITOR_NAMES: list[str] = [f"{p} {s}" for p in COMPETITOR_PREFIXES for s in COMPETITOR_SUFFIXES]

# "Brightline2" removed in the WP3 review (a digit, and a near-copy of the
# Halvard exam competitor "Brightline Medical"). The 19 hand entries split
# 13 / 5 / 1 across train / validation / test, so every test world would
# have sold the same product line; widened with the hand list's own heads
# and tails (no new vocabulary), disjoint from every other pool.
_PRODUCT_LINE_HAND = [
    "Current", "Meraline", "Streamline", "Corewell", "Trueflow", "Beacon",
    "Farview", "Alderline", "Northstar", "Clearpath", "Steadyflow",
    "Trueline", "Farlight", "Coreflow", "Meadowline", "Sterlingflow",
    "Wayfinder", "Lodestar", "Fernline",
]  # fmt: skip
_PRODUCT_LINE_HEADS = [
    "Core", "True", "Clear", "Steady", "Far", "North", "Mera", "Alder",
    "Meadow", "Sterling", "Fern", "Lode", "Stream",
]  # fmt: skip
_PRODUCT_LINE_TAILS = ["line", "flow", "well", "star", "path", "light", "view"]

PRODUCT_LINE_NAMES: list[str] = _combinatorial_prefixes(
    _PRODUCT_LINE_HAND,
    _PRODUCT_LINE_HEADS,
    _PRODUCT_LINE_TAILS,
    exclude=_OTHER_NAME_KEYS
    | {p.lower() for p in VENDOR_PREFIXES}
    | {p.lower() for p in COMPETITOR_PREFIXES},
)


# --------------------------------------------------------------------------
# Contamination name guard
# --------------------------------------------------------------------------


class NameGuard:
    """The training-set builder's name gate (`portolan.training.contamination.
    check_blocklisted_names` over `normalize_words`, the words the gate reads),
    applied to a candidate name before a world uses it. A world built with a
    guard redraws any name, domain, email or handle the gate would block, so
    its records are never excluded for a generated name (the WP7 mini world
    lost a record to the facility qualifier "Parkview")."""

    def __init__(
        self,
        blocklist: dict[str, dict[int, set[int]]],
        max_name_words: int,
        fingerprint: str = "",
    ) -> None:
        self.blocklist = blocklist
        self.max_name_words = max_name_words
        self.fingerprint = fingerprint

    def blocked(self, *names: str) -> bool:
        from portolan.training.contamination import check_blocklisted_names, normalize_words

        return any(
            check_blocklisted_names(normalize_words(name), self.blocklist, self.max_name_words)
            for name in names
            if name
        )

    @classmethod
    def from_dir(cls, contamination_dir: Path) -> NameGuard | None:
        """The guard for the name blocklist at `contamination_dir` (the
        builder's `var/contamination/`), or `None` when none is built there."""
        from portolan.training.contamination import load_name_blocklist

        loaded = load_name_blocklist(contamination_dir)
        if loaded is None:
            return None
        blocklist, params, digest = loaded
        return cls(blocklist, params.max_name_words, digest)
