"""Review-queue volume on a small fixture graph (HANDOFF-2026-09-19,
2026-09-20 06:22: 150 open rows on a small synthetic graph, 140 dismissed
by hand). Every fact and sentence below is shaped on a row of that queue's
backup, one or more per measured cause:

1. the registry never held the vendor itself or its branches, and refused a
   departed contact or churned account even in a source from before they
   left ("Halvard Support" alone was 31 of the 55 unknown-name rows);
2. two undated claims from one source called a contradiction;
3. the resolved-language rule firing on words that assert no resolution;
4. one row per source for the same unknown name.

The same flood run against main at 54c1f40 left 25 open rows (14 unknown
names, 3 possible contradictions, 8 rejected report sentences);
`test_the_flood_leaves_only_the_rows_a_person_must_answer` pinned the 6 left
after that fix, row by row; 4 are open now, since an unknown name that blocks
no fact waits as `watching` until it recurs (migration 0039)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace

import psycopg

from portolan.extract.schema import Coverage, ExtractionResult, Fact, FactType
from portolan.ingest.models import Participant
from portolan.llm.fake import FakeLLMClient
from portolan.merge.pipeline import merge_extraction
from portolan.pipeline import _maybe_import_registry
from portolan.registry import Registry
from portolan.report import weekly
from portolan.report.executive import EXECUTIVE_SYSTEM, build_executive_summary
from portolan.report.models import (
    ActionsFeed,
    CoverageSummary,
    DiffFeed,
    QuietAccounts,
    ThemeReport,
    ThemeSummary,
    WavesFeed,
)
from portolan.search.embed import FakeEmbedder
from tests.merge_helpers import ACCOUNTS_CSV, CONTACTS_CSV, REPO_ROOT, make_source_row

SINCE = date(2026, 8, 31)
AS_OF = date(2026, 9, 6)


@dataclass
class _CorpusConnector:
    """Just enough of `FileConnector` for `_maybe_import_registry`: the CRM
    export, and a root whose parent holds the meta block."""

    root = REPO_ROOT / "corpus" / "out"

    def crm_paths(self):
        return SimpleNamespace(accounts=ACCOUNTS_CSV, contacts=CONTACTS_CSV)


def _fact(fid: str, ftype: FactType, claim: str, entities: list[str], **kw) -> Fact:
    return Fact(
        id=fid,
        type=ftype,
        claim=claim,
        entities=entities,
        locator=kw.pop("locator", "p1"),
        quote=kw.pop("quote", claim),
        confidence=0.9,
        sentiment=0,
        **kw,
    )


def _person(email: str) -> list[Participant]:
    return [Participant(name=email.split("@")[0], email=email)]


# (source slug, type, date, participants, facts). Undated unless a fact
# says otherwise, as most extracted facts are.
SOURCES = [
    (
        "S-ticket-v01",
        "ticket",
        date(2026, 8, 10),
        _person("dhollis@northlakehs.org"),
        [
            _fact(
                "f1",
                FactType.QUESTION,
                "Halvard Support asked for an update on the ticket.",
                ["Halvard Support"],
            ),
            _fact(
                "f2",
                FactType.REQUEST,
                "Derek Hollis asked for documentation on the Tidewater T2 self-test.",
                ["Derek Hollis", "Halvard Support"],
                locator="p2",
            ),
        ],
    ),
    (
        "S-ticket-v02",
        "ticket",
        date(2026, 3, 20),
        _person("nvale@vireosurgical.com"),
        [
            _fact(
                "f1",
                FactType.REQUEST,
                "Nina Vale asked where the self-test documentation is located.",
                ["Nina Vale"],
            ),
            _fact(
                "f2",
                FactType.OTHER,
                "Halvard Support provided the location of the self-test documentation.",
                ["Halvard Support"],
                locator="p2",
            ),
        ],
    ),
    (
        "S-ticket-v03",
        "ticket",
        date(2026, 4, 2),
        _person("tridley@cobaltridgemed.org"),
        [
            _fact(
                "f1",
                FactType.COMPLAINT,
                "Tom Ridley is locked out of the portal after too many failed logins.",
                ["Tom Ridley"],
            ),
            _fact(
                "f2",
                FactType.COMMITMENT,
                "Halvard Support sent a reset link to Tom Ridley's account email.",
                ["Halvard Support", "Tom Ridley"],
                locator="p2",
            ),
        ],
    ),
    (
        "S-crm-v04",
        "crm",
        date(2026, 9, 11),
        [],
        [
            _fact(
                "f1",
                FactType.CONTACT_CHANGE,
                "Tom Ridley left Cobalt Ridge Medical Center on 2026-07-01.",
                ["Tom Ridley", "Cobalt Ridge Medical Center"],
                valid_from=date(2026, 7, 1),
            ),
        ],
    ),
    (
        "S-email-v05",
        "email",
        date(2026, 2, 10),
        [],
        [
            _fact(
                "f1",
                FactType.DECISION,
                "Kestrel Medical Group decided not to renew and moved to Brightline Medical.",
                ["Kestrel Medical Group", "Brightline Medical"],
                valid_from=date(2026, 2, 1),
            ),
        ],
    ),
    (
        "S-call-v06",
        "call",
        date(2026, 8, 5),
        _person("farden@larkspurhealth.org"),
        [
            _fact(
                "f1",
                FactType.PRAISE,
                "Larkspur Memorial's biomed lead called Halvard's field engineer response the "
                "fastest they have had.",
                ["Larkspur Memorial Hospital", "Halvard"],
            ),
        ],
    ),
    (
        "S-email-v07",
        "email",
        date(2026, 8, 4),
        [
            Participant(name="Luis Ferreira", email="l.ferreira@halvardmedical.com"),
            *_person("dhollis@northlakehs.org"),
        ],
        [
            _fact(
                "f1",
                FactType.OTHER,
                "Columbus switched line-haul carriers for outbound Tidewater consumables.",
                ["Columbus"],
                valid_from=date(2026, 8, 3),
            ),
        ],
    ),
    (
        "S-rss-v08",
        "rss",
        date(2026, 9, 2),
        [],
        [
            _fact(
                "f1",
                FactType.CONTACT_CHANGE,
                "Larkspur Regional Health announced Renata Solis as its new Chief Financial "
                "Officer.",
                ["Larkspur Regional Health", "Renata Solis"],
            ),
        ],
    ),
    (
        "S-rss-v09",
        "rss",
        date(2026, 8, 26),
        [],
        [
            _fact(
                "f1",
                FactType.OTHER,
                "Halvard Medical brought on two more drivers out of Dayton.",
                ["Halvard Medical", "Dayton"],
            ),
        ],
    ),
    (
        "S-rss-v10",
        "rss",
        date(2026, 6, 2),
        [],
        [
            _fact(
                "f1",
                FactType.COMPETITOR_MENTION,
                "Riverside Clinic switched infusion pump vendors to Corvane Systems.",
                ["Riverside Clinic", "Corvane Systems"],
            ),
        ],
    ),
    (
        "S-rss-v11",
        "rss",
        date(2026, 6, 9),
        [],
        [
            _fact(
                "f1",
                FactType.COMPETITOR_MENTION,
                "Riverside Clinic finished its Corvane Systems pump rollout.",
                ["Riverside Clinic", "Corvane Systems"],
            ),
        ],
    ),
    (
        "S-call-v12",
        "call",
        date(2026, 7, 30),
        _person("jpark@meadowbrookch.org"),
        [
            _fact(
                "f1",
                FactType.OTHER,
                "June Park said the pharmacy wants Tidewater deliveries confirmed by Tuesday.",
                ["June Park"],
            ),
            _fact(
                "f2",
                FactType.OTHER,
                "June Park said the pharmacy wants Tidewater deliveries confirmed by Wednesday.",
                ["June Park"],
                locator="p2",
            ),
            _fact(
                "f3",
                FactType.OTHER,
                "June Park said Tidewater deliveries to the pharmacy are still arriving late.",
                ["June Park"],
                locator="p3",
            ),
        ],
    ),
    (
        "S-email-v13",
        "email",
        date(2026, 9, 4),
        _person("jpark@meadowbrookch.org"),
        [
            _fact(
                "f1",
                FactType.OTHER,
                "June Park said Tidewater deliveries to the pharmacy seem to be behind us.",
                ["June Park"],
                locator="p3",
                quote="the delivery issues from August seem to be behind us",
            ),
        ],
    ),
]


def _theme(name: str, movement: str) -> ThemeSummary:
    from portolan.registry.text import slugify

    return ThemeSummary(
        slug=f"theme-{slugify(name)}",
        name=name,
        signal_count=3,
        linked_arr=1000.0,
        first_seen=date(2026, 8, 3),
        last_seen=date(2026, 9, 4),
        movement=movement,
    )


THEMES = ThemeReport(
    since=SINCE,
    as_of=AS_OF,
    themes=[
        _theme("Brightline Medical Competitive Push", "resolved"),
        _theme("Quick Turnaround Praise", "resolved"),
        _theme("Tidewater T2 Delivery Delays", "resolution proposed"),
        _theme("Larkspur Biomed Engineering Praise", "rising"),
    ],
)

# Report prose the old rule rejected (live rows 171 to 221), plus two that
# must still be rejected: an unbacked "resolved" and an uncited quote.
EXECUTIVE_SENTENCES = [
    "Tidewater deliveries to Meadowbrook are something to resolve before the pilot begins.",
    "Four actions remain open with none resolved this week.",
    "Meadowbrook's pharmacy director said the August delivery issues 'seem to be behind us' "
    "^[S-email-v13 @ p3].",
    "The Tidewater T2 Delivery Delays theme is only marked 'resolution proposed,' not "
    "confirmed resolved.",
    "Tidewater T2 Delivery Delays is now resolved across the Midwest.",
    "Meadowbrook's pharmacy director says the delivery issues 'seem to be behind us'.",
]
HEADLINE_SENTENCES = [
    "Quick Turnaround Praise and Brightline Medical Competitive Push both closed out this week.",
    "The Wrenfield integration timeline still needs to resolve.",
]


def _responder(*, system, user, json_schema, model, surface):
    del json_schema, model
    if surface == "merge":
        # The compare model calls every June Park pair a contradiction, the
        # way the live rows on her note read; any other pair extends.
        verdict = "contradicts" if "June Park" in user else "extends"
        return {"verdict": verdict, "reason": "the two statements differ"}
    if system == weekly.HEADLINE_SYSTEM:
        return {"sentences": list(HEADLINE_SENTENCES)}
    assert system == EXECUTIVE_SYSTEM
    return {
        "this_week": EXECUTIVE_SENTENCES[:2],
        "attention": EXECUTIVE_SENTENCES[2:4],
        "by_account": [],
        "suggested_actions": EXECUTIVE_SENTENCES[4:],
    }


def build_flood(db: psycopg.Connection) -> None:
    _maybe_import_registry(db, _CorpusConnector())
    registry = Registry.load(db)
    llm = FakeLLMClient(_responder)
    embedder = FakeEmbedder()
    for slug, source_type, source_date, participants, facts in SOURCES:
        source = make_source_row(
            db, slug, source_type=source_type, source_date=source_date, participants=participants
        )
        result = ExtractionResult(
            source_id=slug,
            summary="",
            facts=facts,
            coverage=Coverage(segments_read=1, segments_total=1),
        )
        merge_extraction(db, llm, embedder, registry, result, source)

    diff = DiffFeed(since=SINCE, as_of=AS_OF, quiet_accounts=QuietAccounts(as_of=AS_OF))
    weekly._headline(db, llm, diff, THEMES, [], [])
    build_executive_summary(
        db,
        llm,
        since=SINCE,
        as_of=AS_OF,
        diff=diff,
        themes=THEMES,
        waves=WavesFeed(since=SINCE, as_of=AS_OF),
        questions=[],
        outliers=[],
        contact_changes=[],
        actions=ActionsFeed(),
        coverage=CoverageSummary(since=SINCE, as_of=AS_OF),
    )


def open_rows(db: psycopg.Connection) -> list[tuple[str, str, str]]:
    """(kind, reason or mention, detail) for every open row."""
    with db.cursor() as cur:
        cur.execute(
            "SELECT kind, payload FROM review_queue WHERE status = 'open' ORDER BY id",
        )
        rows = cur.fetchall()
    out = []
    for kind, payload in rows:
        label = payload.get("mention") or payload.get("reason") or ""
        detail = payload.get("detail") or payload.get("claim") or ""
        out.append((kind, label, detail))
    return out


def test_the_flood_leaves_only_the_rows_a_person_must_answer(db: psycopg.Connection) -> None:
    build_flood(db)
    rows = open_rows(db)
    by_label = Counter((kind, label) for kind, label, _detail in rows)

    assert by_label == Counter(
        {
            # The vendor's own news, with no customer to file it under.
            ("other", "no_home_note"): 1,
            # Late in July, "behind us" in September: two sources, undated.
            (
                "other",
                "contradicts verdict downgraded to extends: not both explicitly dated, "
                "or not strictly later",
            ): 1,
            # An unbacked "resolved", and a customer quote with no citation.
            ("other", "resolved_language_without_a_resolved_theme"): 2,
        }
    ), rows
    assert len(rows) == 4

    # Two unknown names on facts that were filed anyway, seen in one and two
    # sources: kept, not yet asked about (`registry.park_mention`).
    with db.cursor() as cur:
        cur.execute(
            "SELECT payload ->> 'mention' FROM review_queue WHERE status = 'watching' ORDER BY 1"
        )
        assert [r[0] for r in cur.fetchall()] == ["Renata Solis", "Riverside Clinic"]


def test_the_second_source_naming_an_unknown_name_joins_its_row(db: psycopg.Connection) -> None:
    build_flood(db)
    with db.cursor() as cur:
        cur.execute(
            "SELECT source_slug, payload -> 'occurrences' FROM review_queue "
            "WHERE kind = 'unknown_entity' AND payload ->> 'mention' = 'Riverside Clinic'"
        )
        (first_source, occurrences) = cur.fetchone()
    assert first_source == "S-rss-v10"
    assert [o["source_slug"] for o in occurrences] == ["S-rss-v11"]
    assert "Corvane" in occurrences[0]["context"]["claim"]


def test_no_fact_is_lost(db: psycopg.Connection) -> None:
    """Every fact the flood merged is a claim, or waits in an open row."""
    build_flood(db)
    with db.cursor() as cur:
        cur.execute("SELECT claim_text FROM claims")
        claims = " ".join(r[0] for r in cur.fetchall())
    for text in (
        "asked for an update on the ticket",
        "provided the location of the self-test documentation",
        "Nina Vale",
        "locked out of the portal",
        "on 2026-07-01",
        "decided not to renew",
        "field engineer response",
        "switched line-haul carriers",
        "Renata Solis",
    ):
        assert text in claims, text
    parked = " ".join(detail for _k, _l, detail in open_rows(db))
    assert "two more drivers out of Dayton" in parked
