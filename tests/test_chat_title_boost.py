"""A chat question's answer notes must reach the prompt even when many other
notes share one of its words in their titles.

The graph below reproduces demo question 7 ("Which accounts have asked for
Tidewater Link to work with Epic?", `scripts/demo_questions.txt`): the
answer notes and claim texts are the live graph's (claims 209, 427, 430 and
280 on 2026-09-23), the notes' descriptions and syntheses follow the live
graph's patterns, and every distractor's title names the product family
"Tidewater", as the live graph's product and product-signal notes do. A
flat partial title boost (a quarter of the top fused score for any one
question word in a title) lifted every distractor above the answer notes,
so none reached the prompt and the answer was "Nothing recorded".
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import ClassVar, NamedTuple

import pytest

from portolan.llm.fake import FakeLLMClient
from portolan.notes.models import Claim, NoteRow, NoteType
from portolan.notes.persist import write_note
from portolan.search.embed import FakeEmbedder
from portolan.search.query import TITLE_BOOST_ALL, _title_boost, search
from portolan.web.answer import NOTHING_RECORDED_MESSAGE, answer_question

QUESTION = "Which accounts have asked for Tidewater Link to work with Epic?"


class Spec(NamedTuple):
    slug: str
    type: NoteType
    title: str
    description: str
    synthesis: str
    related: list[str]
    # (claim_text, source_slug, locator)
    claims: list[tuple[str, str, str]]


def _signal(kind: str, entity_slug: str, entity: str, claims, related=()) -> Spec:
    title = f"{kind.replace('_', ' ').title()} — {entity}"
    return Spec(
        slug=f"signal-{kind}-{entity_slug}",
        type=NoteType.SIGNAL,
        title=title,
        description=f"{kind.replace('_', ' ').title()} signals for [[{entity_slug}]].",
        synthesis=f"{title} tracks one fact type across its linked account.",
        related=[entity_slug, *related],
        claims=list(claims),
    )


def _product(slug: str, title: str) -> Spec:
    return Spec(
        slug=slug,
        type=NoteType.PRODUCT,
        title=title,
        description="A product Halvard sells.",
        synthesis=f"{title} is a product Halvard sells.",
        related=[],
        claims=[],
    )


ST_ALDRIC = _signal(
    "request",
    "st-aldric-health",
    "St. Aldric Health",
    [
        (
            "[[st-aldric-health]] requested a Spanish-language version of the Tidewater T2 "
            "quick-start card.",
            "S-email-021",
            "S-email-021@halvardmedical.com#p2",
        ),
        (
            "[[st-aldric-health]] asked if the [[tidewater-link]] module could talk to their "
            "Epic instance.",
            "S-call-015",
            "09:00:42",
        ),
        (
            "[[st-aldric-health]] requested a firm timeline for the Tidewater Link Epic "
            "integration.",
            "S-email-066",
            "S-email-066@halvardmedical.com#p2",
        ),
    ],
    related=["tidewater-link"],
)
DRIFTWOOD = _signal(
    "request",
    "driftwood-community-hospital",
    "Driftwood Community Hospital",
    [
        (
            "[[driftwood-community-hospital]] requested a review of its bulk-consumables "
            "reorder threshold. ([[carlos-nunez]])",
            "S-ticket-046",
            "S-ticket-046#1",
        ),
        (
            "[[carlos-nunez]] requests a [[tidewater-link]] connection to "
            "[[driftwood-community-hospital]]'s Epic instance.",
            "S-email-068",
            "S-email-068@halvardmedical.com#p2",
        ),
    ],
    related=["tidewater-link"],
)
WRENFIELD = _signal(
    "commitment",
    "wrenfield-health",
    "Wrenfield Health",
    [
        (
            "Priya Natarajan committed to providing a [[tidewater-link]]/Epic integration "
            "test plan by September 10, 2026.",
            "S-call-020",
            "09:14:05",
        ),
    ],
)

TIDEWATER_TITLED = [
    _product("tidewater", "Tidewater"),
    _product("tidewater-link", "Tidewater Link"),
    _product("tidewater-t2", "Tidewater T2"),
    _product("tidewater-t3", "Tidewater T3"),
    _signal(
        "other",
        "tidewater",
        "Tidewater",
        [
            (
                "Columbus switched line-haul carriers for outbound [[tidewater]] consumables "
                "on August 3rd.",
                "S-email-062",
                "S-email-062@halvardmedical.com#p2",
            )
        ],
    ),
    _signal(
        "complaint",
        "tidewater-t2",
        "Tidewater T2",
        [
            (
                "Two shipments of [[tidewater-t2]] consumables showed up days late during the "
                "week of July twenty-seventh.",
                "S-call-007",
                "09:01:51",
            )
        ],
    ),
    _signal(
        "commitment",
        "tidewater-t2",
        "Tidewater T2",
        [
            (
                "Sam Okafor will open a field engineering case to investigate the false "
                "alarms. ([[tidewater-t2]])",
                "S-ticket-024",
                "S-ticket-024#4",
            )
        ],
    ),
    _signal(
        "praise",
        "tidewater-t2",
        "Tidewater T2",
        [
            (
                "The pharmacy team praised the dosing accuracy of the [[tidewater-t2]].",
                "S-call-010",
                "09:03:12",
            )
        ],
    ),
    _signal(
        "decision",
        "tidewater-t3",
        "Tidewater T3",
        [
            (
                "The emergency department agreed to pilot two [[tidewater-t3]] pumps this autumn.",
                "S-call-012",
                "09:05:40",
            )
        ],
    ),
    _signal(
        "usage",
        "tidewater-t2",
        "Tidewater T2",
        [
            (
                "Eighteen [[tidewater-t2]] pumps run across the ICU and med-surg floors.",
                "S-call-016",
                "09:04:10",
            )
        ],
    ),
]

# Other accounts' signals, sharing none of the question's rarer words: they
# give the keyword leg a realistic document frequency for each word.
FILLER_ACCOUNTS = [
    "Cobalt Ridge Medical Center",
    "Anchor Bay Surgical Partners",
    "Larkspur North Hospital",
    "Sable Valley Health",
    "Kestrel Medical Group",
    "Vireo Surgical Group",
    "Northlake Health System",
    "Ashgrove Medical Center",
    "Wrenfield East Hospital",
    "Larkspur Memorial Hospital",
    "Harbor Point Clinic",
    "Juniper Falls Hospital",
]
FILLERS = [
    _signal(
        kind,
        account.lower().replace(" ", "-"),
        account,
        [(text.format(account=account), f"S-ticket-f{i:02d}", f"S-ticket-f{i:02d}#{n}")],
    )
    for i, account in enumerate(FILLER_ACCOUNTS)
    for n, (kind, text) in enumerate(
        (
            ("complaint", "{account} reported a late consumables shipment from Dayton."),
            ("pricing", "{account} expects a credit on its next service invoice."),
        ),
        start=1,
    )
]

GRAPH = [ST_ALDRIC, DRIFTWOOD, WRENFIELD, *TIDEWATER_TITLED, *FILLERS]


class _WordSetEmbedder(FakeEmbedder):
    """`FakeEmbedder` over each distinct word once, "asked"/"requests"/
    "requested" read as "request": closer to how the live graph's sentence
    embeddings ranked its notes (the answer notes near the top) than raw
    word counts, which favour short title chunks that repeat a product name."""

    _SAME: ClassVar[dict[str, str]] = {
        "asked": "request",
        "requests": "request",
        "requested": "request",
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        words = [
            dict.fromkeys(self._SAME.get(w, w) for w in re.findall(r"[a-z0-9]+", t.lower()))
            for t in texts
        ]
        return super().embed([" ".join(w) for w in words])


def _write(conn, embedder, spec: Spec) -> list[int]:
    now = datetime.now(UTC)
    note = NoteRow(
        slug=spec.slug,
        type=spec.type,
        title=spec.title,
        description=spec.description,
        created=now,
        updated=now,
        synthesis=spec.synthesis,
        related=spec.related,
    )
    write_note(conn, note, [], changed_by="test", reason="fixture")
    claims = []
    with conn.cursor() as cur:
        for text, source_slug, locator in spec.claims:
            cur.execute(
                """
                INSERT INTO sources (slug, source_type, s3_key, content_hash, source_date,
                                     extraction_status)
                VALUES (%s, 'call', %s, %s, '2026-08-21', 'done')
                ON CONFLICT (slug) DO NOTHING
                """,
                (source_slug, f"originals/call/{source_slug}", f"hash-{source_slug}"),
            )
            cur.execute(
                """
                INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator,
                                    quote, recorded)
                VALUES (%s, 'request', %s, %s, %s, %s, %s) RETURNING id, recorded
                """,
                (spec.slug, text, source_slug, locator, text, now),
            )
            claim_id, recorded = cur.fetchone()
            claims.append(
                Claim(
                    id=claim_id,
                    note_slug=spec.slug,
                    fact_type="request",
                    claim_text=text,
                    source_slug=source_slug,
                    locator=locator,
                    quote=text,
                    recorded=recorded,
                )
            )
    conn.commit()
    write_note(conn, note, claims, changed_by="test", reason="fixture", embedder=embedder)
    return [c.id for c in claims]


@pytest.fixture()
def epic_graph(db):
    embedder = _WordSetEmbedder()
    claim_ids = {spec.slug: _write(db, embedder, spec) for spec in GRAPH}
    return db, embedder, claim_ids


_CLAIM_LINE_RE = re.compile(r"^- \[claim_id=(\d+)\] (.*)$")


def _grounded_model(**kwargs) -> dict:
    """A stand-in for the chat model that answers only from the prompt: one
    sentence per supplied claim that mentions Epic, cited to that claim and
    its note; with none, the one uncited "not answered" sentence the system
    prompt asks for."""
    sentences = []
    note_slug = None
    for line in kwargs["user"].splitlines():
        if line.startswith("### Note "):
            note_slug = line.removeprefix("### Note ").strip()
            continue
        match = _CLAIM_LINE_RE.match(line)
        if match and "Epic" in match.group(2):
            sentences.append(
                {
                    "text": match.group(2).split(" (source:")[0],
                    "note_slugs": [note_slug],
                    "claim_ids": [int(match.group(1))],
                }
            )
    if not sentences:
        sentences = [{"text": "The notes do not say.", "note_slugs": [], "claim_ids": []}]
    return {"sentences": sentences}


def test_answer_notes_reach_the_top_eight_over_notes_sharing_one_title_word(epic_graph):
    conn, embedder, _claim_ids = epic_graph
    slugs = [h.note_slug for h in search(conn, embedder, QUESTION, k=8).hits]
    assert ST_ALDRIC.slug in slugs
    assert DRIFTWOOD.slug in slugs


def test_demo_question_7_is_answered_with_the_request_claims(epic_graph):
    conn, embedder, claim_ids = epic_graph
    answer = answer_question(
        conn, FakeLLMClient(_grounded_model), embedder, QUESTION, k=8, expand=1
    )

    assert answer.sentences[0].text != NOTHING_RECORDED_MESSAGE
    cited = {c.claim_id for s in answer.sentences for c in s.citations}
    _card, st_aldric_ask, st_aldric_timeline = claim_ids[ST_ALDRIC.slug]
    _threshold, driftwood_ask = claim_ids[DRIFTWOOD.slug]
    assert {st_aldric_ask, st_aldric_timeline, driftwood_ask} <= cited


def test_one_question_word_in_a_title_earns_its_share_of_the_title_boost():
    terms = ["which", "accounts", "have", "asked", "tidewater", "link", "work", "epic"]
    assert _title_boost(terms, "Tidewater T2", 1.0) == pytest.approx(TITLE_BOOST_ALL / 8)
    assert _title_boost(terms, "Tidewater Link", 1.0) == pytest.approx(TITLE_BOOST_ALL / 4)
    assert _title_boost(["ashgrove", "health"], "Ashgrove Health", 1.0) == TITLE_BOOST_ALL
    assert _title_boost(["ashgrove", "renewal"], "Ashgrove Health", 1.0) == pytest.approx(
        TITLE_BOOST_ALL / 2
    )
