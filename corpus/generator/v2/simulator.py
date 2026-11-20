"""`simulate_world()` -- the v2 world builder (`01-spec/world-generator-v2-
2026-09-22.md` section 1). Deterministic in `(pack.id, seed, months,
split)`: same inputs, same `World`, byte for byte.

**How it builds a world**, mirroring the spec's "Timeline" paragraph:

1. Draw a vendor identity, an account roster, and a people roster --
   every name from `v2.names`' split-partitioned pools (`split` chooses
   which sub-pool; `seed` chooses which entries of it).
2. For each account, draw one or two arcs from the pack's arc library
   (weighted, without replacement) and instantiate them into a chain of
   dated `Event`s, each one causally linked to the step before it
   (`Event.causes`). Every event emits one `Fact` per fact type its
   `EventTemplate` names -- code-written claims, never model-invented
   (spec section 1: "No fact is ever model-invented").
3. Top up: the spec's per-type floor (section 1: "at least 25 facts of
   each of the 12 types per world") is met the way v1's
   `balanced_type_mix` meets it -- by adding more events, of whichever
   type still emits the short type, until every `FactType` clears 25.
4. Weekly state snapshots (spec: "The simulator steps week by week ...
   keeps a state snapshot per date") record, per account, who holds each
   role, current ARR, open incidents and renewal status.

Every `Fact` traces to a real `Event.id` by construction here; `World`'s
own `model_validator` (`v2.models.World._facts_trace_to_events`) re-checks
that in case a caller builds a `World` from pieces some other way.
"""

from __future__ import annotations

import calendar
import hashlib
import itertools
import random
from collections import Counter
from collections.abc import Callable
from datetime import date, timedelta

from corpus.generator.v2 import names
from corpus.generator.v2.models import (
    Account,
    AccountKind,
    AccountState,
    AccountStatus,
    AccountTier,
    Arc,
    Competitor,
    Employment,
    Event,
    EventType,
    Fact,
    HealthPoint,
    Person,
    Persona,
    PersonSide,
    Product,
    Release,
    RenewalStatus,
    Split,
    StateSnapshot,
    Vendor,
    World,
)
from corpus.generator.v2.packs.schema import (
    ArcTemplate,
    EventTemplate,
    PersonaTemplate,
    VerticalPack,
)
from portolan.extract.schema import FactType

FACT_TYPE_FLOOR = 25  # spec section 1: "at least 25 facts of each of the 12 types per world"

REGIONS = ["Midwest", "Northeast", "South", "West"]

_VENDOR_ROLE_POOL = [
    "account rep",
    "account rep",
    "customer success lead",
    "operations manager",
    "product manager",
    "support engineer",
]

_REFERENCE_MODE_WEIGHTS = [
    ("explicit", 0.75),
    ("alias", 0.06),
    ("first_name", 0.06),
    ("title", 0.05),
    ("handle", 0.04),
    ("pronoun", 0.04),
]


def _slug(text: str) -> str:
    return text.lower().replace(" ", "-").replace(".", "").replace("'", "")


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _clamp_date(d: date, lo: date, hi: date) -> date:
    return min(max(d, lo), hi)


def _weighted_sample_without_replacement(
    rng: random.Random, items: list, weights: list[float], k: int
) -> list:
    """`k` distinct items from `items`, weighted, no replacement -- Python
    stdlib's `random.choices` allows repeats, so this does the classic
    "weighted draw, then remove" loop instead."""
    pool = list(items)
    pool_weights = list(weights)
    chosen = []
    for _ in range(min(k, len(pool))):
        pick = rng.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pool.pop(idx))
        pool_weights.pop(idx)
    return chosen


def _build_persona(rng: random.Random, template: PersonaTemplate) -> Persona:
    lo, hi = template.typo_rate_range
    typo_rate = round(rng.uniform(lo, hi), 3) if hi > lo else lo
    return Persona(
        speech_register=template.speech_register,
        formality=template.formality,
        typo_rate=typo_rate,
        chat_style=template.chat_style,
        signature=template.signature_style,
        hedging=template.hedging,
        second_language_features=list(template.second_language_features),
    )


def _seed_index(pack: VerticalPack, seed: int, salt: str, pool_size: int) -> int:
    """Deterministic pool index for this world's vendor or competitor.
    `(offset + seed) % size`, so two worlds of the same vertical with
    different seeds below the pool size never draw the same entry (spec
    section 4: "Every world has its own vendor, products and
    competitors"); a plain `rng.choice` let two worlds of one split share a
    vendor."""
    offset = int(hashlib.sha256(f"{salt}:{pack.id}".encode()).hexdigest(), 16)
    return (offset + seed) % pool_size


def _scatter(pool: list[str], salt: str) -> list[str]:
    """`pool` in a fixed hash order, so neighbouring indices are unrelated
    names (the composed pools list "Hemlockford", "Hemlockhall", ...
    side by side, and a world drew two competitors that close)."""
    return sorted(pool, key=lambda name: hashlib.sha256(f"{salt}:{name}".encode()).hexdigest())


_REDRAWS = 40


def _ok(guard: names.NameGuard | None, *candidates: str) -> bool:
    return guard is None or not guard.blocked(*candidates)


def _redraw(
    draw: Callable[[], str],
    guard: names.NameGuard | None,
    derived: Callable[[str], list[str]] = lambda _name: [],
) -> str | None:
    """`draw()` until the name, and the strings `derived` builds from it,
    are ones the contamination gate would not block. The first draw is kept
    when nothing is blocked, so a world without a guard is unchanged."""
    for _ in range(_REDRAWS):
        name = draw()
        if _ok(guard, name, *derived(name)):
            return name
    return None


def _product_names(pack: VerticalPack, line: str) -> list[str]:
    return [
        archetype.name_template.format(line=line, tier=tier).strip()
        for archetype in pack.products
        for tier in (archetype.tiers or [""])
    ]


def _build_vendor(
    rng: random.Random,
    pack: VerticalPack,
    split: Split,
    seed: int,
    guard: names.NameGuard | None = None,
) -> Vendor:
    # Split by prefix (the part documents repeat as the short name, alias
    # and email domain), not by the full composed name -- WP3 review; see
    # `names.vendor_prefixes_for_split`.
    vendor_prefixes = _scatter(names.vendor_prefixes_for_split(split), "vendor")
    line_pool = names.pool_for_split(names.PRODUCT_LINE_NAMES, split, "product_line")
    competitor_prefixes = _scatter(names.competitor_prefixes_for_split(split), "competitor")
    tld = pack.name_pool_refs.vendor_domain_tld

    start = _seed_index(pack, seed, "vendor", len(vendor_prefixes))
    vendor_short, vendor_name = "", None
    for step in range(len(vendor_prefixes)):
        vendor_short = vendor_prefixes[(start + step) % len(vendor_prefixes)]
        if not _ok(guard, vendor_short, f"{_slug(vendor_short)}{tld}"):
            continue
        vendor_name = _redraw(
            lambda s=vendor_short: f"{s} {rng.choice(names.VENDOR_SUFFIXES)}", guard
        )
        if vendor_name is not None:
            break
    if vendor_name is None:
        raise ValueError("name guard: every vendor name of this split is blocklisted")
    domain = f"{_slug(vendor_short)}{tld}"
    line = _redraw(lambda: rng.choice(line_pool), guard, lambda x: _product_names(pack, x))
    if line is None:
        raise ValueError("name guard: no product line of this split passes the blocklist")

    products: list[Product] = []
    for archetype in pack.products:
        tiers = archetype.tiers or [""]
        for tier in tiers:
            product_name = archetype.name_template.format(line=line, tier=tier).strip()
            product_id = f"product-{_slug(product_name)}"
            releases = [
                Release(
                    version=f"{rng.randint(1, 5)}.{rng.randint(0, 9)}",
                    date=date(2025, rng.randint(1, 12), rng.randint(1, 28)),
                    changes=["stability and performance improvements"],
                )
                for _ in range(rng.randint(1, 3))
            ]
            known_issues = rng.sample(
                archetype.known_issue_pool,
                k=min(len(archetype.known_issue_pool), rng.randint(0, 2)),
            )
            products.append(
                Product(
                    id=product_id,
                    name=product_name,
                    aliases=[tier] if tier else [],
                    kind=archetype.kind,
                    releases=releases,
                    known_issues=known_issues,
                )
            )

    first = _seed_index(pack, seed * 2, "competitor", len(competitor_prefixes))
    competitor_names: list[str] = []
    for k in range(len(competitor_prefixes)):
        if len(competitor_names) >= min(2, len(competitor_prefixes)):
            break
        prefix = competitor_prefixes[(first + k) % len(competitor_prefixes)]
        if not _ok(guard, prefix):
            continue
        name = _redraw(lambda p=prefix: f"{p} {rng.choice(names.COMPETITOR_SUFFIXES)}", guard)
        if name is not None:
            competitor_names.append(name)
    competitors = [
        Competitor(
            id=f"competitor-{_slug(name)}",
            name=name,
            aliases=[name.split()[0]],
            positioning="a lower-cost alternative in the same category",
        )
        for name in competitor_names
    ]

    return Vendor(
        id=f"vendor-{_slug(vendor_short)}",
        name=vendor_name,
        vertical=pack.id,
        domain=domain,
        message_id_domain=domain,
        overview=f"{vendor_name} sells {pack.vendor_what}.",
        initiatives=[],
        products=products,
        competitors=competitors,
    )


def _account_domain(stem: str) -> str:
    return f"{_slug(stem)}health.org"


def _build_accounts(
    rng: random.Random,
    pack: VerticalPack,
    split: Split,
    window_start: date,
    guard: names.NameGuard | None = None,
) -> list[Account]:
    stem_pool = names.pool_for_split(names.ACCOUNT_STEMS, split, "account_stem")
    num_accounts = rng.randint(8, 16)
    stems = rng.sample(stem_pool, k=min(num_accounts, len(stem_pool)))
    if guard is not None:
        spare = [s for s in stem_pool if s not in stems]
        for idx, stem in enumerate(stems):
            while not _ok(guard, stem, _account_domain(stem)):
                if not spare:
                    raise ValueError("name guard: the account stems of this split run out")
                stem = spare.pop(rng.randrange(len(spare)))
            stems[idx] = stem

    def _named(stem: str, suffixes: list[str]) -> str:
        name = _redraw(lambda: f"{stem} {rng.choice(suffixes)}", guard)
        if name is None:
            raise ValueError(f"name guard: every name for the stem {stem!r} is blocklisted")
        return name

    accounts: list[Account] = []
    crm_counter = itertools.count(4001)
    i = 0
    while len(accounts) < num_accounts and i < len(stems):
        stem = stems[i]
        i += 1
        tier = rng.choices(
            [AccountTier.ENTERPRISE, AccountTier.MID_MARKET, AccountTier.SMB],
            weights=[0.25, 0.45, 0.30],
            k=1,
        )[0]
        arr_range = {
            AccountTier.ENTERPRISE: (800_000, 3_000_000),
            AccountTier.MID_MARKET: (200_000, 800_000),
            AccountTier.SMB: (50_000, 200_000),
        }[tier]
        region = rng.choice(REGIONS)
        contract_start = window_start - timedelta(days=rng.randint(30, 730))
        renewal_date = contract_start + timedelta(days=365 * rng.randint(1, 3))
        has_children = rng.random() < 0.25 and (num_accounts - len(accounts)) >= 2

        domain = _account_domain(stem)
        if has_children:
            parent_name = _named(stem, pack.name_pool_refs.account_suffixes_system)
            parent_id = f"acct-{_slug(stem)}"
            parent = Account(
                id=parent_id,
                crm_id=f"A-{next(crm_counter)}",
                name=parent_name,
                kind=AccountKind.SYSTEM,
                parent=None,
                tier=tier,
                arr=0.0,
                contract_start=contract_start,
                renewal_date=renewal_date,
                region=region,
                domain=domain,
                aliases=[stem],
            )
            accounts.append(parent)
            n_children = 2 if (num_accounts - len(accounts)) >= 2 else 1
            # Distinct (qualifier, suffix) pairs, drawn without replacement
            # from the combinatorial pool -- never a numeric suffix (WP3
            # review, defect 3; `names.FACILITY_QUALIFIERS`).
            facility_pool = [
                f"{stem} {qualifier} {suffix}"
                for qualifier in names.FACILITY_QUALIFIERS
                for suffix in pack.name_pool_refs.account_suffixes_facility
            ]
            facility_names = rng.sample(facility_pool, k=n_children)
            if guard is not None:
                spare = [n for n in facility_pool if n not in facility_names and _ok(guard, n)]
                for idx, name in enumerate(facility_names):
                    if not _ok(guard, name):
                        facility_names[idx] = spare.pop(rng.randrange(len(spare)))
            for c in range(n_children):
                if len(accounts) >= num_accounts:
                    break
                lo, hi = arr_range
                child_name = facility_names[c]
                child = Account(
                    id=f"acct-{_slug(child_name)}",
                    crm_id=f"A-{next(crm_counter)}",
                    name=child_name,
                    kind=AccountKind.FACILITY,
                    parent=parent_id,
                    tier=tier,
                    arr=float(rng.randint(lo, hi) // (c + 2)),
                    contract_start=contract_start,
                    renewal_date=renewal_date,
                    region=region,
                    domain=domain,
                    aliases=[stem],
                )
                accounts.append(child)
        else:
            kind = rng.choice([AccountKind.FACILITY, AccountKind.STANDALONE])
            suffixes = (
                pack.name_pool_refs.account_suffixes_facility
                if kind == AccountKind.FACILITY
                else pack.name_pool_refs.account_suffixes_standalone
            )
            lo, hi = arr_range
            account = Account(
                id=f"acct-{_slug(stem)}",
                crm_id=f"A-{next(crm_counter)}",
                name=_named(stem, suffixes),
                kind=kind,
                parent=None,
                tier=tier,
                arr=float(rng.randint(lo, hi)),
                contract_start=contract_start,
                renewal_date=renewal_date,
                region=region,
                domain=domain,
                aliases=[stem],
            )
            accounts.append(account)

    # Deterministic health history: one point roughly monthly for six months.
    for account in accounts:
        account.health = [
            HealthPoint(date=window_start + timedelta(days=30 * k), score=rng.randint(55, 95))
            for k in range(6)
        ]
    return accounts[:num_accounts]


def _build_people(
    rng: random.Random,
    pack: VerticalPack,
    vendor: Vendor,
    accounts: list[Account],
    split: Split,
    window_start: date,
    guard: names.NameGuard | None = None,
) -> list[Person]:
    first_pool = names.pool_for_split(names.FIRST_NAMES, split, "first_name")
    last_pool = names.pool_for_split(names.LAST_NAMES, split, "last_name")
    used_names: set[str] = set()

    def _usable(first: str, last: str) -> bool:
        # The name, the email local part and the chat handle built below.
        return f"{first} {last}" not in used_names and _ok(
            guard, f"{first} {last}", f"{first[0]}{_slug(last)}", f"{first}{last.lower()[:3]}"
        )

    def _draw_name() -> tuple[str, str]:
        for _ in range(30):
            first = rng.choice(first_pool)
            last = rng.choice(last_pool)
            full = f"{first} {last}"
            if _usable(first, last):
                used_names.add(full)
                return first, last
        # Thirty random misses: walk the full first x last product from a
        # random start instead. The old fallback built a numbered name
        # ("... 417"), recorded it, and then returned the plain (first,
        # last) pair anyway -- a duplicate person name and person id.
        start = rng.randrange(len(first_pool) * len(last_pool))
        total = len(first_pool) * len(last_pool)
        for step in range(total):
            idx = (start + step) % total
            first, last = first_pool[idx // len(last_pool)], last_pool[idx % len(last_pool)]
            full = f"{first} {last}"
            if _usable(first, last):
                used_names.add(full)
                return first, last
        raise ValueError("name pool exhausted for this split")  # pragma: no cover

    people: list[Person] = []

    n_vendor_staff = rng.randint(10, 14)
    for _ in range(n_vendor_staff):
        first, last = _draw_name()
        role = rng.choice(_VENDOR_ROLE_POOL)
        persona_template = rng.choice(pack.persona_bank)
        person_id = f"person-{_slug(first)}-{_slug(last)}"
        people.append(
            Person(
                id=person_id,
                name=f"{first} {last}",
                side=PersonSide.VENDOR,
                account_id=None,
                role=role,
                title=role.title(),
                email=f"{first[0].lower()}.{last.lower()}@{vendor.domain}",
                chat_handle=f"@{first.lower()}{last.lower()[:3]}",
                tenure=Employment(
                    start=window_start - timedelta(days=rng.randint(60, 1200)), end=None
                ),
                persona=_build_persona(rng, persona_template),
            )
        )

    for account in accounts:
        if account.kind == AccountKind.SYSTEM:
            continue  # parent system accounts hold no direct contacts
        n_contacts = rng.randint(3, 6)
        for _ in range(n_contacts):
            first, last = _draw_name()
            buyer_role = rng.choice(pack.roles)
            persona_template = rng.choice(pack.persona_bank)
            person_id = f"person-{_slug(first)}-{_slug(last)}"
            tenure_start = window_start - timedelta(days=rng.randint(0, 900))
            # ~12% turnover over the window (spec: "10 to 15% contact
            # turnover over the window").
            turns_over = rng.random() < 0.12
            tenure_end = window_start + timedelta(days=rng.randint(14, 200)) if turns_over else None
            person = Person(
                id=person_id,
                name=f"{first} {last}",
                side=PersonSide.CUSTOMER,
                account_id=account.id,
                role=buyer_role.title,
                title=buyer_role.title,
                email=f"{first[0].lower()}{_slug(last)}@{account.domain}",
                chat_handle=f"@{first.lower()}{last.lower()[:3]}",
                tenure=Employment(start=tenure_start, end=tenure_end),
                persona=_build_persona(rng, persona_template),
            )
            people.append(person)
            if turns_over and tenure_end is not None:
                r_first, r_last = _draw_name()
                replacement_id = f"person-{_slug(r_first)}-{_slug(r_last)}"
                replacement = Person(
                    id=replacement_id,
                    name=f"{r_first} {r_last}",
                    side=PersonSide.CUSTOMER,
                    account_id=account.id,
                    role=buyer_role.title,
                    title=buyer_role.title,
                    email=f"{r_first[0].lower()}{_slug(r_last)}@{account.domain}",
                    chat_handle=f"@{r_first.lower()}{r_last.lower()[:3]}",
                    tenure=Employment(start=tenure_end + timedelta(days=1), end=None),
                    persona=_build_persona(rng, rng.choice(pack.persona_bank)),
                )
                people.append(replacement)
                # `person` is a pydantic model already appended to `people`
                # by value; mutate the list entry in place via replacement.
                people[-2] = person.model_copy(update={"replaced_by": replacement_id})

    return people


def _render_claim(
    rng: random.Random,
    template: str,
    *,
    account: Account,
    vendor: Vendor,
    product: Product | None,
    competitor: Competitor | None,
    person: Person | None,
    event_date: date,
) -> tuple[str, list[str], list[str]]:
    """Renders one claim template. Returns `(claim, entities, anchors)`."""
    ctx = {
        "account": account.name,
        "vendor": vendor.name,
        "product": product.name
        if product
        else (vendor.products[0].name if vendor.products else "the product"),
        "competitor": competitor.name if competitor else "a competing vendor",
        "person": person.name if person else "the account team",
        "date": event_date.isoformat(),
        "amount": f"{rng.randint(2, 20) * 5}%",
        "days": str(rng.randint(3, 21)),
    }
    claim = template.format(**ctx)

    entities = [account.id]
    anchors = [account.name]
    if product is not None and "{product}" in template:
        entities.append(product.id)
        anchors.append(product.name)
    if competitor is not None and "{competitor}" in template:
        entities.append(competitor.id)
        anchors.append(competitor.name)
    if person is not None and "{person}" in template:
        entities.append(person.id)
        anchors.append(person.name)
    return claim, entities, anchors


def _pick_reference_mode(rng: random.Random) -> str:
    modes = [m for m, _ in _REFERENCE_MODE_WEIGHTS]
    weights = [w for _, w in _REFERENCE_MODE_WEIGHTS]
    return rng.choices(modes, weights=weights, k=1)[0]


class _WorldBuilder:
    """Mutable scratch state for one `simulate_world()` call -- keeps the
    id counters and lookups in one place instead of threading eight
    positional arguments through every helper."""

    def __init__(
        self,
        rng: random.Random,
        pack: VerticalPack,
        vendor: Vendor,
        accounts: list[Account],
        people: list[Person],
    ):
        self.rng = rng
        self.pack = pack
        self.vendor = vendor
        self.accounts = accounts
        self.people = people
        self.events: list[Event] = []
        self.facts: list[Fact] = []
        self.arcs: list[Arc] = []
        self._event_ids = itertools.count(1)
        self._fact_ids = itertools.count(1)
        self._arc_ids = itertools.count(1)
        self.customer_people_by_account: dict[str, list[Person]] = {}
        for p in people:
            if p.side == PersonSide.CUSTOMER and p.account_id:
                self.customer_people_by_account.setdefault(p.account_id, []).append(p)
        self.vendor_people = [p for p in people if p.side == PersonSide.VENDOR]
        self.templates_by_type: dict[EventType, EventTemplate] = {
            t.event_type: t for t in pack.event_templates
        }

    def next_event_id(self) -> str:
        return f"E{next(self._event_ids):05d}"

    def next_fact_id(self) -> str:
        return f"F{next(self._fact_ids):05d}"

    def next_arc_id(self) -> str:
        return f"ARC{next(self._arc_ids):04d}"

    def active_contacts(self, account: Account, on: date) -> list[Person]:
        """Customer contacts of `account` employed on `on` (WP3 review:
        speakers used to be drawn from every contact ever on the roster, so
        19 to 34 facts per world were spoken by someone who had already
        left or had not yet joined -- the spec's "a document dated D sees
        only the state at D" failed at the speaker). Falls back to the
        whole roster only if nobody is employed that day."""
        contacts = self.customer_people_by_account.get(account.id) or []
        active = [
            p
            for p in contacts
            if p.tenure.start <= on and (p.tenure.end is None or p.tenure.end >= on)
        ]
        return active or contacts

    def speaker_for(
        self,
        account: Account,
        fact_type: FactType,
        on: date,
        exclude: Person | None = None,
    ) -> Person | None:
        if fact_type == FactType.COMMITMENT and self.vendor_people:
            return self.rng.choice(self.vendor_people)
        contacts = self.active_contacts(account, on)
        if exclude is not None and len(contacts) > 1:
            contacts = [p for p in contacts if p.id != exclude.id]
        return self.rng.choice(contacts) if contacts else None

    def emit_event_and_facts(
        self,
        *,
        account: Account,
        event_type: EventType,
        event_date: date,
        causes: list[str],
    ) -> Event:
        template = self.templates_by_type[event_type]
        product = self.rng.choice(self.vendor.products) if self.vendor.products else None
        competitor = (
            self.rng.choice(self.vendor.competitors)
            if event_type == EventType.COMPETITIVE_EVAL and self.vendor.competitors
            else None
        )
        contacts = self.active_contacts(account, event_date)
        champion_candidates = [
            p
            for p in contacts
            if any(r.title == p.title and r.is_champion_candidate for r in self.pack.roles)
        ] or contacts
        event_person = self.rng.choice(champion_candidates) if champion_candidates else None

        event = Event(
            id=self.next_event_id(),
            date=event_date,
            type=event_type,
            account_id=account.id,
            product_ids=[product.id] if product else [],
            people_ids=[p.id for p in ([event_person] if event_person else []) + contacts[:1]],
            causes=causes,
            state_changes=[],
        )
        self.events.append(event)

        for fact_type, templates in template.claim_templates.items():
            claim_template = self.rng.choice(templates)
            names_person = "{person}" in claim_template
            speaker = self.speaker_for(
                account, fact_type, event_date, exclude=event_person if names_person else None
            )
            person_for_claim = event_person if names_person else speaker
            claim, entities, anchors = _render_claim(
                self.rng,
                claim_template,
                account=account,
                vendor=self.vendor,
                product=product,
                competitor=competitor,
                person=person_for_claim,
                event_date=event_date,
            )
            fact = Fact(
                id=self.next_fact_id(),
                fact_type=fact_type,
                claim=claim,
                entities=entities,
                speaker=speaker.id if speaker else "customer",
                date=event_date,
                account_id=account.id,
                event_id=event.id,
                anchors=anchors,
                directive=self.rng.choice(self.pack.fact_directives[fact_type]),
                reference_mode=_pick_reference_mode(self.rng),
            )
            self.facts.append(fact)
        return event

    def emit_arc(
        self, account: Account, arc_template: ArcTemplate, window_start: date, window_end: date
    ) -> Arc:
        window_days = (window_end - window_start).days
        last_offset = arc_template.steps[-1].day_offset
        latest_start_offset = max(1, window_days - last_offset - 1)
        arc_start = window_start + timedelta(days=self.rng.randint(0, latest_start_offset))

        event_ids: list[str] = []
        prev_event_id: str | None = None
        for step in arc_template.steps:
            event_date = _clamp_date(
                arc_start + timedelta(days=step.day_offset), window_start, window_end
            )
            causes = [prev_event_id] if prev_event_id else []
            event = self.emit_event_and_facts(
                account=account, event_type=step.event_type, event_date=event_date, causes=causes
            )
            event_ids.append(event.id)
            prev_event_id = event.id

        arc = Arc(
            id=self.next_arc_id(),
            arc_type=arc_template.arc_type,
            account_id=account.id,
            event_ids=event_ids,
        )
        self.arcs.append(arc)
        return arc

    def top_up_fact_type_floors(
        self, window_start: date, window_end: date, floor: int = FACT_TYPE_FLOOR
    ) -> None:
        window_days = (window_end - window_start).days
        counts = Counter(f.fact_type for f in self.facts)
        candidates_by_type: dict[FactType, list[EventTemplate]] = {ft: [] for ft in FactType}
        for template in self.pack.event_templates:
            for ft in template.claim_templates:
                candidates_by_type[ft].append(template)

        # Leaf accounts only, like the arcs above (WP3 review): a system
        # (parent) account holds no contacts, so every top-up event that
        # landed on one produced facts with speaker "customer" and no
        # person to put in a document -- 66 to 71 facts per world.
        leaf_accounts = [a for a in self.accounts if a.kind != AccountKind.SYSTEM] or self.accounts
        for fact_type in FactType:
            candidates = candidates_by_type[fact_type]
            guard = 0
            while counts[fact_type] < floor and guard < floor * 10:
                guard += 1
                template = self.rng.choice(candidates)
                account = self.rng.choice(leaf_accounts)
                event_date = window_start + timedelta(days=self.rng.randint(0, window_days))
                event = self.emit_event_and_facts(
                    account=account,
                    event_type=template.event_type,
                    event_date=event_date,
                    causes=[],
                )
                # `emit_event_and_facts` emits one fact per fact type the
                # template carries (possibly several); recount from the
                # source of truth rather than assuming +1.
                counts = Counter(f.fact_type for f in self.facts)
                del event


def simulate_world(
    pack: VerticalPack,
    seed: int,
    months: int = 8,
    split: Split = Split.TRAIN,
    *,
    name_guard: names.NameGuard | None = None,
) -> World:
    """Build one deterministic world. Same `(pack.id, seed, months, split)`
    always returns an identical `World` (see `test_world_generator_v2.py::
    test_simulate_world_is_deterministic_for_a_seed`). With `name_guard`,
    every name the contamination gate would block is redrawn: the world is
    then a function of the guard's blocklist too, and is the same world
    whenever the blocklist blocks none of its names."""
    if not (6 <= months <= 12):
        raise ValueError(f"months must be 6-12 per spec section 1, got {months}")

    rng = random.Random(f"{pack.id}:{seed}:{months}:{split.value}")

    window_start = date(2026, 1, 5) + timedelta(weeks=(seed % 8))
    window_end = _add_months(window_start, months) - timedelta(days=1)

    vendor = _build_vendor(rng, pack, split, seed, name_guard)
    accounts = _build_accounts(rng, pack, split, window_start, name_guard)
    people = _build_people(rng, pack, vendor, accounts, split, window_start, name_guard)

    builder = _WorldBuilder(rng, pack, vendor, accounts, people)

    for account in accounts:
        if account.kind == AccountKind.SYSTEM:
            continue  # arcs run against leaf accounts, which carry the ARR/facts
        n_arcs = rng.choice([1, 1, 2])
        weights = [t.weight for t in pack.arcs]
        chosen = _weighted_sample_without_replacement(rng, pack.arcs, weights, n_arcs)
        for arc_template in chosen:
            builder.emit_arc(account, arc_template, window_start, window_end)

    builder.top_up_fact_type_floors(window_start, window_end)

    snapshots = _build_snapshots(accounts, people, builder.events, window_start, window_end)

    return World(
        id=f"world-{pack.id}-{seed}",
        seed=seed,
        vertical=pack.id,
        split=split,
        window_start=window_start,
        window_end=window_end,
        vendor=vendor,
        accounts=accounts,
        people=people,
        events=builder.events,
        facts=builder.facts,
        arcs=builder.arcs,
        snapshots=snapshots,
    )


def _build_snapshots(
    accounts: list[Account],
    people: list[Person],
    events: list[Event],
    window_start: date,
    window_end: date,
) -> list[StateSnapshot]:
    """Spec section 1: "The simulator steps week by week ... keeps a
    state snapshot per date: who holds each role, current ARR, open
    incidents, renewal status." One snapshot per week of the window."""
    events_by_account: dict[str, list[Event]] = {}
    for event in events:
        events_by_account.setdefault(event.account_id, []).append(event)

    people_by_account: dict[str, list[Person]] = {}
    for person in people:
        if person.side == PersonSide.CUSTOMER and person.account_id:
            people_by_account.setdefault(person.account_id, []).append(person)

    snapshots: list[StateSnapshot] = []
    current = window_start
    while current <= window_end:
        account_states: list[AccountState] = []
        for account in accounts:
            role_holders: dict[str, str] = {}
            for person in people_by_account.get(account.id, []):
                if person.tenure.start <= current and (
                    person.tenure.end is None or person.tenure.end >= current
                ):
                    role_holders[person.role] = person.id

            open_incidents = [
                e.id
                for e in events_by_account.get(account.id, [])
                if e.type == EventType.INCIDENT and e.date <= current <= e.date + timedelta(days=30)
            ]

            has_recent_eval = any(
                e.type == EventType.COMPETITIVE_EVAL and abs((e.date - current).days) <= 45
                for e in events_by_account.get(account.id, [])
            )
            days_to_renewal = (account.renewal_date - current).days
            if current > account.renewal_date:
                renewal_status = RenewalStatus.RENEWED
            elif has_recent_eval:
                renewal_status = RenewalStatus.AT_RISK
            elif days_to_renewal <= 60:
                renewal_status = RenewalStatus.IN_PROGRESS
            else:
                renewal_status = RenewalStatus.NOT_DUE
            if account.status == AccountStatus.CHURNED:
                renewal_status = RenewalStatus.CHURNED

            account_states.append(
                AccountState(
                    account_id=account.id,
                    role_holders=role_holders,
                    arr=account.arr,
                    open_incident_event_ids=open_incidents,
                    renewal_status=renewal_status,
                )
            )
        snapshots.append(StateSnapshot(date=current, account_states=account_states))
        current += timedelta(weeks=1)

    return snapshots
