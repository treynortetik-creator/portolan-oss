"""Labeled data for the Jev evaluation (var/jev-typesafe-brief-2026-09-16.md,
task A: theme-resolution judge + three lint gates).

Every entry traces to a fact id in `corpus/truth-table.yaml` unless marked
`fact_id=None`, in which case it is hand-authored (the corpus plants no
speculative or opinion-flavored claims at all, and the resolution/account
mismatches are deliberate constructions, not something extracted from a real
pipeline run -- both are noted per-entry below and totaled in the report).

THEME_A = "Late deliveries, Midwest" (accounts: ashgrove-medical,
larkspur-memorial, cobalt-ridge, northlake-general, meadowbrook).
`resolution_facts: [F61, F62, F63, F64, F65]` is the corpus's own planted
resolving set -- every RESOLUTION_PAIRS positive is one of those five, used
verbatim (paraphrased into a two-sentence candidate the way a real extracted
claim reads, never inventing new content).

THEME_B = "Tidewater Link / Epic integration requests" (accounts: st-aldric,
wrenfield, driftwood) never resolves in this corpus -- every pair against it
is a negative, several deliberately hard (same product ask, wrong account or
wrong date; a commitment or unanswered question that sounds close to closure
but isn't).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ThemeFixture:
    slug: str
    name: str
    # (account_slug, member_claim_text)
    members: list[tuple[str, str]]


THEME_A = ThemeFixture(
    slug="theme-late-deliveries-eval",
    name="Late deliveries, Midwest",
    members=[
        (
            "ashgrove-medical",
            "Ashgrove Medical Center received two late Tidewater T2 consumable "
            "deliveries in the week of 2026-07-27; two shipments showed up days late.",
        ),
        (
            "larkspur-memorial",
            "Larkspur Memorial Hospital reported a Tidewater T2 consumable delivery "
            "arriving four days late on 2026-08-04.",
        ),
        (
            "meadowbrook",
            "Meadowbrook Community Hospital's Tidewater T2 consumable order of "
            "2026-08-12 arrived five days late.",
        ),
    ],
)

THEME_B = ThemeFixture(
    slug="theme-epic-integration-eval",
    name="Tidewater Link / Epic integration requests",
    members=[
        (
            "st-aldric",
            "St. Aldric Health asked whether Tidewater Link could integrate with its "
            "Epic instance, the first of three such requests.",
        ),
        (
            "wrenfield",
            "Wrenfield Health reversed course on 2026-08-24 and asked Halvard to begin "
            "Tidewater Link integration with its Epic instance immediately.",
        ),
        (
            "driftwood",
            "Driftwood Community Hospital's pharmacy director asked on 2026-08-27 for a "
            "Tidewater Link connection to its Epic instance, the third of three such "
            "requests.",
        ),
    ],
)


@dataclass(frozen=True)
class ResolutionPair:
    pair_id: str
    fact_id: str | None
    theme: ThemeFixture
    fact_type: str
    account: str | None
    claim_date: date
    sentences: list[str]
    label: bool  # ground truth: does this candidate resolve the theme's problem?
    resolving_sentence: int | None  # 1-based index into `sentences`, positives only
    category: str
    hand_authored: bool = False


RESOLUTION_PAIRS: list[ResolutionPair] = [
    # ---------------- Positives: the corpus's own planted resolution_facts ----------------
    ResolutionPair(
        "P01",
        "F62",
        THEME_A,
        "praise",
        "ashgrove-medical",
        date(2026, 8, 31),
        [
            "The last two Tidewater T2 deliveries came in right on schedule.",
            "Ashgrove Medical Center's ICU nurse manager mentioned this while wrapping up the call.",
        ],
        True,
        1,
        "positive",
    ),
    ResolutionPair(
        "P02",
        "F63",
        THEME_A,
        "praise",
        "larkspur-memorial",
        date(2026, 9, 2),
        [
            "Larkspur Memorial Hospital's biomed lead followed up after the August delivery problems.",
            "Deliveries had been on time for two straight weeks and he was dropping the second-supplier idea for now.",
        ],
        True,
        2,
        "positive",
    ),
    ResolutionPair(
        "P03",
        "F64",
        THEME_A,
        "praise",
        "cobalt-ridge",
        date(2026, 9, 3),
        [
            "Its last Tidewater T2 order showed up exactly when promised.",
            "Cobalt Ridge Medical Center's materials manager mentioned this during a routine check-in.",
        ],
        True,
        1,
        "positive",
    ),
    ResolutionPair(
        "P04",
        "F65",
        THEME_A,
        "praise",
        "meadowbrook",
        date(2026, 9, 4),
        [
            "Meadowbrook Community Hospital's pharmacy director gave an update on the delivery situation.",
            "The delivery issues from August seem to be behind us now.",
        ],
        True,
        2,
        "positive",
    ),
    ResolutionPair(
        "P05",
        "F61",
        THEME_A,
        "decision",
        None,
        date(2026, 8, 31),
        [
            "The Dayton distribution branch hired two additional drivers, effective 2026-08-25, "
            "after weeks of being short-staffed.",
            "Halvard's operations manager confirmed this on an internal call.",
        ],
        True,
        1,
        "positive_hard_no_named_account",
    ),
    # ---------------- Negatives, Theme A: mentions the problem, doesn't resolve it ----------------
    ResolutionPair(
        "N01",
        "F01",
        THEME_A,
        "complaint",
        "ashgrove-medical",
        date(2026, 7, 30),
        [
            "Rob Tallis called in about a recurring shipping problem this week.",
            "Two Tidewater T2 consumable deliveries showed up days late during the week of 2026-07-27.",
        ],
        False,
        None,
        "negative_mentions_problem",
    ),
    ResolutionPair(
        "N02",
        "F03",
        THEME_A,
        "complaint",
        "cobalt-ridge",
        date(2026, 8, 7),
        [
            "Aisha Bell emailed about a recent supply shortfall.",
            "A Tidewater T2 delivery arrived three days late on 2026-08-06 and the hospital had to "
            "borrow consumables from Larkspur Memorial.",
        ],
        False,
        None,
        "negative_mentions_problem",
    ),
    ResolutionPair(
        "N03",
        "F04",
        THEME_A,
        "complaint",
        "northlake-general",
        date(2026, 8, 14),
        [
            "Derek Hollis described two separate delivery incidents on the call.",
            "A Tidewater T3 delivery arrived two days late on 2026-08-11 and a second one three days "
            "late on 2026-08-13.",
        ],
        False,
        None,
        "negative_mentions_problem",
    ),
    ResolutionPair(
        "N04",
        "F06",
        THEME_A,
        "risk",
        "ashgrove-medical",
        date(2026, 8, 20),
        [
            "Ashgrove Health's CFO raised the delivery issue as a renewal concern.",
            "She said continued late deliveries would be raised at the 2027 renewal.",
        ],
        False,
        None,
        "negative_hard_risk_not_resolution",
    ),
    ResolutionPair(
        "N05",
        "F09",
        THEME_A,
        "commitment",
        "ashgrove-medical",
        date(2026, 8, 20),
        [
            "Sam Okafor followed up on the outstanding delivery issue.",
            "He promised Ashgrove Health a written delivery root-cause summary by 2026-08-28.",
        ],
        False,
        None,
        "negative_hard_commitment_not_resolution",
    ),
    ResolutionPair(
        "N06",
        "F37",
        THEME_A,
        "complaint",
        "cobalt-ridge",
        date(2026, 8, 11),
        [
            "Aisha Bell flagged another delivery miss this month.",
            "Cobalt Ridge Medical Center's second Tidewater T2 delivery of the month was four days "
            "overdue on 2026-08-11.",
        ],
        False,
        None,
        "negative_mentions_problem",
    ),
    ResolutionPair(
        "N07",
        "F44",
        THEME_A,
        "complaint",
        "ashgrove-riverside",
        date(2026, 8, 19),
        [
            "Maria Chen reported a fresh delivery complaint.",
            "Ashgrove Riverside Hospital's Tidewater T2 consumable delivery arrived five days late on "
            "2026-08-18.",
        ],
        False,
        None,
        "negative_hard_different_account_same_family",
    ),
    ResolutionPair(
        "N08",
        "F47",
        THEME_A,
        "risk",
        "larkspur-memorial",
        date(2026, 8, 19),
        [
            "Felix Arden warned about the ongoing delivery problems.",
            "He said if deliveries did not stabilize by September he would recommend Larkspur pilot a "
            "second consumables supplier.",
        ],
        False,
        None,
        "negative_hard_conditional_not_resolution",
    ),
    ResolutionPair(
        "N09",
        "F48",
        THEME_A,
        "commitment",
        "larkspur-memorial",
        date(2026, 8, 19),
        [
            "Sam Okafor scheduled a review of the delivery root cause.",
            "He committed to a joint call with Halvard's operations team and Larkspur Memorial by "
            "2026-08-27.",
        ],
        False,
        None,
        "negative_hard_commitment_not_resolution",
    ),
    ResolutionPair(
        "N10",
        "F51",
        THEME_A,
        "complaint",
        "cobalt-ridge",
        date(2026, 8, 22),
        [
            "Aisha Bell reported yet another late shipment.",
            "Cobalt Ridge Medical Center's second late Tidewater T2 delivery in three weeks arrived on "
            "2026-08-22, four days behind schedule.",
        ],
        False,
        None,
        "negative_mentions_problem",
    ),
    # ---------------- Negatives, Theme A: different problem/account entirely ----------------
    ResolutionPair(
        "N11",
        "F19",
        THEME_A,
        "request",
        "ashgrove-medical",
        date(2026, 7, 30),
        [
            "Rob Tallis raised a product feature request during the call.",
            "Ashgrove Medical Center's ICU asked for a night-mode alarm volume on the Tidewater T3.",
        ],
        False,
        None,
        "negative_hard_same_account_different_topic",
    ),
    ResolutionPair(
        "N12",
        "F23",
        THEME_A,
        "complaint",
        "larkspur-north",
        date(2026, 7, 9),
        [
            "Felix Arden reported a device alarm issue.",
            "Larkspur North Hospital reported Tidewater T2 occlusion alarms triggering falsely on "
            "three pumps in 2026-07.",
        ],
        False,
        None,
        "negative_hard_different_account_different_problem",
    ),
    ResolutionPair(
        "N13",
        "F13",
        THEME_A,
        "pricing",
        "st-aldric",
        date(2026, 5, 12),
        [
            "Owen Castellane disputed a recent invoice.",
            "St. Aldric Health's 2026 HalvardCare invoice of 2026-05-01 was disputed for a duplicated "
            "line item of 4,800 dollars.",
        ],
        False,
        None,
        "negative_different_account",
    ),
    ResolutionPair(
        "N14",
        "F16",
        THEME_A,
        "competitor_mention",
        "vireo",
        date(2026, 5, 20),
        [
            "Nina Vale mentioned a competing vendor's pitch.",
            "Vireo Surgical Group received a Brightline Medical proposal in 2026-05 and considered it "
            "15 percent cheaper than Halvard's.",
        ],
        False,
        None,
        "negative_different_account",
    ),
    ResolutionPair(
        "N15",
        "F27",
        THEME_A,
        "usage",
        "vireo",
        date(2026, 5, 20),
        [
            "Nina Vale described current pump deployment.",
            "Vireo Surgical Group uses Tidewater T2 pumps in two operating rooms and none in recovery "
            "as of 2026-05-20.",
        ],
        False,
        None,
        "negative_different_account",
    ),
    ResolutionPair(
        "N16",
        "F34",
        THEME_A,
        "decision",
        "kestrel",
        date(2026, 3, 4),
        [
            "Priya Natarajan noted an account status change.",
            "Kestrel Medical Group ended its Halvard relationship in 2026-02 after moving to Brightline "
            "Medical.",
        ],
        False,
        None,
        "negative_different_account",
    ),
    ResolutionPair(
        "N17",
        "F14",
        THEME_A,
        "risk",
        "sable-valley",
        date(2026, 7, 8),
        [
            "Paul Strand described a budget constraint.",
            "Sable Valley Health froze new capital equipment purchases from 2026-07-01 through the "
            "end of 2026.",
        ],
        False,
        None,
        "negative_different_account",
    ),
    ResolutionPair(
        "N18",
        "F24",
        THEME_A,
        "praise",
        "sable-valley",
        date(2026, 7, 8),
        [
            "Paul Strand praised a recent product integration.",
            "He called Tidewater Link's HL7 feed flawless since go-live on 2026-07-08.",
        ],
        False,
        None,
        "negative_hard_praise_wrong_topic",
    ),
    ResolutionPair(
        "N19",
        "F29",
        THEME_A,
        "contact_change",
        "larkspur",
        date(2026, 9, 2),
        [
            "An industry feed reported a leadership change.",
            "Larkspur Regional Health appointed a new Chief Financial Officer, effective 2026-09-01.",
        ],
        False,
        None,
        "negative_different_account",
    ),
    ResolutionPair(
        "N20",
        "F30",
        THEME_A,
        "contact_change",
        "ashgrove",
        date(2026, 8, 24),
        [
            "Maria Chen emailed from her personal address about a role change.",
            "She moved from Director of Procurement to Director of Clinical Operations at Ashgrove "
            "Health in 2026-08.",
        ],
        False,
        None,
        "negative_different_account",
    ),
    ResolutionPair(
        "N31",
        "F35",
        THEME_A,
        "other",
        "ashgrove",
        date(2026, 9, 1),
        [
            "The CRM export recorded updated account financials.",
            "Ashgrove Health's annual recurring revenue with Halvard is 2,350,000 dollars as of "
            "2026-09-01.",
        ],
        False,
        None,
        "negative_pure_distractor",
    ),
    ResolutionPair(
        "N32",
        "F36",
        THEME_A,
        "other",
        "larkspur",
        date(2026, 3, 2),
        [
            "The CRM export recorded an ownership relationship.",
            "Larkspur Regional Health is owned by Pinecrest Health Partners.",
        ],
        False,
        None,
        "negative_pure_distractor",
    ),
    ResolutionPair(
        "N35",
        "F70",
        THEME_A,
        "competitor_mention",
        "anchor-bay",
        date(2026, 8, 15),
        [
            "Renee Castillo mentioned a competitor outreach.",
            "A Brightline Medical rep had reached out to Anchor Bay Surgical Partners but they haven't "
            "taken a meeting.",
        ],
        False,
        None,
        "negative_different_account",
    ),
    # ---------------- Negatives, Theme B: never resolves in this corpus ----------------
    ResolutionPair(
        "N21",
        "F50",
        THEME_B,
        "request",
        "st-aldric",
        date(2026, 8, 21),
        [
            "Owen Castellane followed up again about the Epic integration timeline.",
            "St. Aldric Health followed up on 2026-08-21 asking for a firm timeline on the Tidewater "
            "Link Epic integration, the second of three such requests.",
        ],
        False,
        None,
        "negative_hard_repeated_ask_not_resolution",
    ),
    ResolutionPair(
        "N22",
        "F40",
        THEME_B,
        "request",
        "wrenfield",
        date(2026, 8, 10),
        [
            "Yolanda Pierce discussed the Epic integration timeline.",
            "Wrenfield Health told Halvard it would defer any Tidewater Link/Epic integration "
            "discussion until 2027, effectively asking to shelve the request.",
        ],
        False,
        None,
        "negative_hard_deferral_not_resolution",
    ),
    ResolutionPair(
        "N23",
        "F57",
        THEME_B,
        "commitment",
        "wrenfield",
        date(2026, 8, 24),
        [
            "Priya Natarajan committed to next steps on the integration.",
            "She promised Wrenfield Health a Tidewater Link/Epic integration test plan by 2026-09-10.",
        ],
        False,
        None,
        "negative_hard_commitment_not_resolution",
    ),
    ResolutionPair(
        "N24",
        "F58",
        THEME_B,
        "commitment",
        "driftwood",
        date(2026, 8, 28),
        [
            "Dana Whitlock committed to a follow-up on the integration project.",
            "She committed to sending Driftwood Community Hospital a formal Tidewater Link project "
            "timeline by 2026-09-05.",
        ],
        False,
        None,
        "negative_hard_commitment_not_resolution",
    ),
    ResolutionPair(
        "N25",
        "F67",
        THEME_B,
        "question",
        "wrenfield-east",
        date(2026, 9, 5),
        [
            "Sophie Lund asked about the integration timeline for her own facility.",
            "Wrenfield East Hospital asked whether the Tidewater Link integration timeline applied to "
            "its facility too, and did not receive an answer in the same thread.",
        ],
        False,
        None,
        "negative_hard_different_account_same_family",
    ),
    ResolutionPair(
        "N26",
        "F43",
        THEME_B,
        "praise",
        "driftwood",
        date(2026, 8, 13),
        [
            "Carlos Nunez praised the pump's performance on the call.",
            "He called the Tidewater T2's dosing accuracy spot on every time on 2026-08-13.",
        ],
        False,
        None,
        "negative_hard_praise_wrong_topic_same_account",
    ),
    ResolutionPair(
        "N27",
        "F42",
        THEME_B,
        "competitor_mention",
        "driftwood",
        date(2026, 8, 13),
        [
            "Carlos Nunez mentioned a past vendor comparison.",
            "Driftwood Community Hospital said it had briefly compared Corvane Systems pumps last "
            "year but it never went anywhere.",
        ],
        False,
        None,
        "negative_different_topic",
    ),
    ResolutionPair(
        "N28",
        "F71",
        THEME_B,
        "request",
        "driftwood",
        date(2026, 8, 6),
        [
            "Carlos Nunez raised a separate operational request.",
            "Driftwood Community Hospital asked for a review of its bulk-consumables reorder "
            "threshold.",
        ],
        False,
        None,
        "negative_hard_same_account_different_topic",
    ),
    ResolutionPair(
        "N29",
        "F20",
        THEME_B,
        "request",
        "larkspur",
        date(2026, 6, 9),
        [
            "Gwen Marsh raised an integration ask months before the current wave started.",
            "Larkspur Regional Health asked for Tidewater Link integration with its Epic instance by "
            "2027-01.",
        ],
        False,
        None,
        "negative_hard_same_ask_different_account",
    ),
    ResolutionPair(
        "N30",
        "F22",
        THEME_B,
        "question",
        "northlake",
        date(2026, 6, 17),
        [
            "Derek Hollis asked about a different integration capability.",
            "Northlake Health System asked whether the Tidewater T3 supports Cerner integration and "
            "did not receive an answer.",
        ],
        False,
        None,
        "negative_hard_similar_ask_different_system",
    ),
    ResolutionPair(
        "N33",
        "F26",
        THEME_B,
        "usage",
        "larkspur-memorial",
        date(2026, 8, 5),
        [
            "Felix Arden described the hospital's current pump fleet.",
            "Larkspur Memorial Hospital has forty Tidewater T2 pumps and is trialling four T3 units "
            "as of 2026-08-05.",
        ],
        False,
        None,
        "negative_wrong_theme_and_account",
    ),
    ResolutionPair(
        "N34",
        "F10",
        THEME_B,
        "praise",
        "larkspur-memorial",
        date(2026, 8, 5),
        [
            "Felix Arden praised Halvard's service response.",
            "He called Halvard's field engineer response to the July pump alarm issue the fastest "
            "we've had from any vendor.",
        ],
        False,
        None,
        "negative_hard_praise_wrong_theme_and_account",
    ),
]

assert len(RESOLUTION_PAIRS) == 40, len(RESOLUTION_PAIRS)
assert sum(p.label for p in RESOLUTION_PAIRS) == 5
assert sum(not p.label for p in RESOLUTION_PAIRS) == 35


# ---------------------------------------------------------------------------
# Track A2: three lint gates, ~60 claims.
#
# Groups (see 01-spec/jev-eval-2026-09-16/A-judge-and-lint.md "Method" for the
# full accounting):
#   G1 (20): attributed=True,  factual=True,  about_account=True  -- straight
#            from the truth table, named speaker, matched account.
#   G2 (10): attributed=False, factual=True,  about_account=True  -- truth-
#            table content, attribution stripped.
#   G3 (10): attributed=True,  factual=False, about_account=True  -- HAND-
#            AUTHORED hedged/speculative phrasing around a truth-table topic
#            (the corpus plants no opinions) -- hand_authored=True.
#   G4 (10): attributed=True,  factual=True,  about_account=False -- truth-
#            table content, deliberately paired with the WRONG account.
#   G5 (10): attributed=False, factual=False, about_account=mixed -- HAND-
#            AUTHORED, no source and speculative; 5 True / 5 False on account.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintClaim:
    claim_id: str
    fact_id: str | None
    text: str
    presented_account: str
    attributed: bool
    factual: bool
    about_account: bool
    hand_authored: bool = False


LINT_CLAIMS: list[LintClaim] = [
    # ---- G1: attributed, factual, correct account (20) ----
    LintClaim(
        "L01",
        "F01",
        "Rob Tallis, the ICU nurse manager at Ashgrove Medical Center, said two "
        "Tidewater T2 consumable deliveries showed up days late the week of 2026-07-27.",
        "Ashgrove Medical Center",
        True,
        True,
        True,
    ),
    LintClaim(
        "L02",
        "F02",
        "Felix Arden, Larkspur Memorial Hospital's biomed engineering lead, reported "
        "a Tidewater T2 delivery arriving four days late on 2026-08-04.",
        "Larkspur Memorial Hospital",
        True,
        True,
        True,
    ),
    LintClaim(
        "L03",
        "F09",
        "Sam Okafor, the Halvard account rep, promised Ashgrove Health a written "
        "delivery root-cause summary by 2026-08-28.",
        "Ashgrove Health",
        True,
        True,
        True,
    ),
    LintClaim(
        "L04",
        "F13",
        "Owen Castellane, St. Aldric Health's VP of Operations, disputed a 4,800 "
        "dollar duplicated line item on the 2026-05-01 HalvardCare invoice.",
        "St. Aldric Health",
        True,
        True,
        True,
    ),
    LintClaim(
        "L05",
        "F19",
        "Rob Tallis asked for a night-mode alarm volume setting on the Tidewater T3 "
        "for Ashgrove Medical Center's ICU.",
        "Ashgrove Medical Center",
        True,
        True,
        True,
    ),
    LintClaim(
        "L06",
        "F20",
        "Gwen Marsh, Larkspur Regional Health's VP of Supply Chain, asked for "
        "Tidewater Link integration with its Epic instance by 2027-01.",
        "Larkspur Regional Health",
        True,
        True,
        True,
    ),
    LintClaim(
        "L07",
        "F28",
        "Aisha Bell told Halvard she was taking over from Tom Ridley as Cobalt Ridge "
        "Medical Center's Materials Manager.",
        "Cobalt Ridge Medical Center",
        True,
        True,
        True,
    ),
    LintClaim(
        "L08",
        "F31",
        "Gwen Marsh said Larkspur Regional Health decided to standardise both "
        "hospitals on the Tidewater T3 by 2027-06.",
        "Larkspur Regional Health",
        True,
        True,
        True,
    ),
    LintClaim(
        "L09",
        "F33",
        "Aisha Bell agreed that Cobalt Ridge Medical Center would pilot two Tidewater "
        "T3 units in its emergency department starting 2026-10-01.",
        "Cobalt Ridge Medical Center",
        True,
        True,
        True,
    ),
    LintClaim(
        "L10",
        "F41",
        "Yolanda Pierce said Wrenfield Medical Center runs eighteen Tidewater T2 "
        "pumps across its ICU and med-surg floors.",
        "Wrenfield Medical Center",
        True,
        True,
        True,
    ),
    LintClaim(
        "L11",
        "F43",
        "Carlos Nunez, Driftwood Community Hospital's pharmacy director, called the "
        "Tidewater T2's dosing accuracy spot on every time.",
        "Driftwood Community Hospital",
        True,
        True,
        True,
    ),
    LintClaim(
        "L12",
        "F47",
        "Felix Arden warned that if deliveries did not stabilize by September, he "
        "would recommend Larkspur Memorial pilot a second consumables supplier.",
        "Larkspur Memorial Hospital",
        True,
        True,
        True,
    ),
    LintClaim(
        "L13",
        "F52",
        "Aisha Bell said Cobalt Ridge Medical Center expects a delivery-delay credit "
        "on its next HalvardCare invoice after last month's problems.",
        "Cobalt Ridge Medical Center",
        True,
        True,
        True,
    ),
    LintClaim(
        "L14",
        "F54",
        "Renee Castillo, Anchor Bay Surgical Partners' practice administrator, asked "
        "for a small-volume discount on Tidewater T2 consumables, citing budget pressure.",
        "Anchor Bay Surgical Partners",
        True,
        True,
        True,
    ),
    LintClaim(
        "L15",
        "F57",
        "Priya Natarajan, the Halvard account rep, promised Wrenfield Health a "
        "Tidewater Link/Epic integration test plan by 2026-09-10.",
        "Wrenfield Health",
        True,
        True,
        True,
    ),
    LintClaim(
        "L16",
        "F58",
        "Dana Whitlock committed to sending Driftwood Community Hospital a formal "
        "Tidewater Link project timeline by 2026-09-05.",
        "Driftwood Community Hospital",
        True,
        True,
        True,
    ),
    LintClaim(
        "L17",
        "F62",
        "Rob Tallis said the last two Tidewater T2 deliveries to Ashgrove Medical "
        "Center came in right on schedule.",
        "Ashgrove Medical Center",
        True,
        True,
        True,
    ),
    LintClaim(
        "L18",
        "F63",
        "Felix Arden confirmed Larkspur Memorial Hospital's deliveries had been on "
        "time for two straight weeks.",
        "Larkspur Memorial Hospital",
        True,
        True,
        True,
    ),
    LintClaim(
        "L19",
        "F64",
        "Aisha Bell said Cobalt Ridge Medical Center's last Tidewater T2 order showed "
        "up exactly when promised.",
        "Cobalt Ridge Medical Center",
        True,
        True,
        True,
    ),
    LintClaim(
        "L20",
        "F70",
        "Renee Castillo said a Brightline Medical rep had reached out to Anchor Bay "
        "Surgical Partners but they haven't taken a meeting.",
        "Anchor Bay Surgical Partners",
        True,
        True,
        True,
    ),
    # ---- G2: unattributed, factual, correct account (10) ----
    LintClaim(
        "L21",
        "F03",
        "A Tidewater T2 delivery to Cobalt Ridge Medical Center arrived three days "
        "late on 2026-08-06, and the hospital had to borrow consumables from Larkspur Memorial.",
        "Cobalt Ridge Medical Center",
        False,
        True,
        True,
    ),
    LintClaim(
        "L22",
        "F04",
        "Northlake General Hospital received a Tidewater T3 delivery two days late "
        "on 2026-08-11 and a second one three days late on 2026-08-13.",
        "Northlake General Hospital",
        False,
        True,
        True,
    ),
    LintClaim(
        "L23",
        "F14",
        "Sable Valley Health froze new capital equipment purchases from 2026-07-01 "
        "through the end of 2026.",
        "Sable Valley Health",
        False,
        True,
        True,
    ),
    LintClaim(
        "L24",
        "F23",
        "Larkspur North Hospital reported Tidewater T2 occlusion alarms triggering "
        "falsely on three pumps in 2026-07.",
        "Larkspur North Hospital",
        False,
        True,
        True,
    ),
    LintClaim(
        "L25",
        "F29",
        "Larkspur Regional Health appointed a new Chief Financial Officer, effective 2026-09-01.",
        "Larkspur Regional Health",
        False,
        True,
        True,
    ),
    LintClaim(
        "L26",
        "F34",
        "Kestrel Medical Group ended its Halvard relationship in 2026-02 after moving "
        "to Brightline Medical.",
        "Kestrel Medical Group",
        False,
        True,
        True,
    ),
    LintClaim(
        "L27",
        "F44",
        "Ashgrove Riverside Hospital's Tidewater T2 consumable delivery arrived five "
        "days late on 2026-08-18.",
        "Ashgrove Riverside Hospital",
        False,
        True,
        True,
    ),
    LintClaim(
        "L28",
        "F59",
        "Northlake Health System's Derek Hollis was promoted from Director of "
        "Clinical Engineering to VP of Facilities effective 2026-08-24.",
        "Northlake Health System",
        False,
        True,
        True,
    ),
    LintClaim(
        "L29",
        "F66",
        "Anchor Bay Surgical Partners uses six Tidewater T2 pumps across two "
        "operating rooms as of 2026-09-01.",
        "Anchor Bay Surgical Partners",
        False,
        True,
        True,
    ),
    LintClaim(
        "L30",
        "F71",
        "Driftwood Community Hospital asked for a review of its bulk-consumables "
        "reorder threshold.",
        "Driftwood Community Hospital",
        False,
        True,
        True,
    ),
    # ---- G3: attributed, OPINION/SPECULATION (hand-authored), correct account (10) ----
    LintClaim(
        "L31",
        "F06",
        "Helen Ozoro, Ashgrove Health's CFO, thinks continued late deliveries will "
        "probably come up again at the 2027 renewal, though nothing is decided yet.",
        "Ashgrove Health",
        True,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L32",
        "F17",
        "June Park said she suspects Meadowbrook Community Hospital might eventually "
        "look at other pump vendors, but she wasn't sure.",
        "Meadowbrook Community Hospital",
        True,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L33",
        "F22",
        "Derek Hollis guessed that the Tidewater T3 probably supports Cerner "
        "integration, though he wasn't certain.",
        "Northlake Health System",
        True,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L34",
        "F53",
        "Gwen Marsh believes the 12 percent volume discount will likely become "
        "non-negotiable if delivery issues continue.",
        "Larkspur Regional Health",
        True,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L35",
        None,
        "Aisha Bell speculated that Cobalt Ridge Medical Center might reconsider its "
        "HalvardCare renewal if credits don't materialize.",
        "Cobalt Ridge Medical Center",
        True,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L36",
        "F42",
        "Carlos Nunez said he imagines Driftwood Community Hospital could look at "
        "Corvane again someday, though nothing is planned.",
        "Driftwood Community Hospital",
        True,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L37",
        "F67",
        "Sophie Lund wondered whether Wrenfield East Hospital's integration timeline "
        "might slip, but admitted she was only guessing.",
        "Wrenfield East Hospital",
        True,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L38",
        None,
        "Nina Vale said she thinks Vireo Surgical Group will probably expand its "
        "Tidewater T2 fleet eventually, but had no timeline.",
        "Vireo Surgical Group",
        True,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L39",
        "F50",
        "Owen Castellane speculated that the Epic integration for St. Aldric Health "
        "might slip past the original estimate.",
        "St. Aldric Health",
        True,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L40",
        "F57",
        "Priya Natarajan guessed the Tidewater Link test plan for Wrenfield Health "
        "would probably be ready early, but couldn't promise it.",
        "Wrenfield Health",
        True,
        False,
        True,
        hand_authored=True,
    ),
    # ---- G4: attributed, factual, WRONG account (10, deliberately mismatched) ----
    LintClaim(
        "L41",
        "F02",
        "Felix Arden reported a Tidewater T2 delivery arriving four days late on 2026-08-04.",
        "Northlake General Hospital",
        True,
        True,
        False,
    ),
    LintClaim(
        "L42",
        "F44",
        "Maria Chen reported that Ashgrove Riverside Hospital's Tidewater T2 delivery "
        "arrived five days late on 2026-08-18.",
        "Ashgrove Medical Center",
        True,
        True,
        False,
    ),
    LintClaim(
        "L43",
        "F23",
        "Felix Arden reported Tidewater T2 occlusion alarms triggering falsely on "
        "three pumps at Larkspur North Hospital in 2026-07.",
        "Larkspur Memorial Hospital",
        True,
        True,
        False,
    ),
    LintClaim(
        "L44",
        "F54",
        "Renee Castillo asked for a small-volume discount on Tidewater T2 consumables "
        "for Anchor Bay Surgical Partners.",
        "Cobalt Ridge Medical Center",
        True,
        True,
        False,
    ),
    LintClaim(
        "L45",
        "F41",
        "Yolanda Pierce said Wrenfield Medical Center runs eighteen Tidewater T2 "
        "pumps across its ICU and med-surg floors.",
        "Wrenfield East Hospital",
        True,
        True,
        False,
    ),
    LintClaim(
        "L46",
        "F67",
        "Sophie Lund asked whether the Tidewater Link integration timeline for "
        "Wrenfield East Hospital applied to its facility too.",
        "Wrenfield Medical Center",
        True,
        True,
        False,
    ),
    LintClaim(
        "L47",
        "F16",
        "Nina Vale said Vireo Surgical Group received a Brightline Medical proposal "
        "it considered 15 percent cheaper than Halvard's.",
        "Sable Valley Health",
        True,
        True,
        False,
    ),
    LintClaim(
        "L48",
        "F13",
        "Owen Castellane disputed a 4,800 dollar duplicated line item on St. Aldric "
        "Health's 2026-05-01 HalvardCare invoice.",
        "Vireo Surgical Group",
        True,
        True,
        False,
    ),
    LintClaim(
        "L49",
        "F38",
        "Felix Arden reported Larkspur North Hospital's Tidewater T2 consumable "
        "order placed 2026-08-08 arrived three days late.",
        "Larkspur Regional Health",
        True,
        True,
        False,
    ),
    LintClaim(
        "L50",
        "F51",
        "Aisha Bell reported Cobalt Ridge Medical Center's second late Tidewater T2 "
        "delivery in three weeks, arriving four days behind schedule on 2026-08-22.",
        "Northlake General Hospital",
        True,
        True,
        False,
    ),
    # ---- G5: unattributed, OPINION/SPECULATION (hand-authored), account mixed (10) ----
    LintClaim(
        "L51",
        None,
        "Deliveries to Ashgrove Medical Center will probably keep slipping through the fall.",
        "Ashgrove Medical Center",
        False,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L52",
        None,
        "Pricing pressure at Larkspur Regional Health could get worse if the discount "
        "request is denied.",
        "Larkspur Regional Health",
        False,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L53",
        None,
        "The Epic integration for Driftwood Community Hospital might not be finished on time.",
        "Driftwood Community Hospital",
        False,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L54",
        None,
        "Cobalt Ridge Medical Center may eventually look at a second consumables supplier.",
        "Cobalt Ridge Medical Center",
        False,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L55",
        None,
        "Wrenfield Health's integration test plan will probably slip a little.",
        "Wrenfield Health",
        False,
        False,
        True,
        hand_authored=True,
    ),
    LintClaim(
        "L56",
        "F14",
        "Sable Valley Health might unfreeze capital purchases sometime next year.",
        "Ashgrove Health",
        False,
        False,
        False,
        hand_authored=True,
    ),
    LintClaim(
        "L57",
        None,
        "Vireo Surgical Group could switch vendors if pricing doesn't improve.",
        "St. Aldric Health",
        False,
        False,
        False,
        hand_authored=True,
    ),
    LintClaim(
        "L58",
        "F34",
        "Kestrel Medical Group's old account might eventually be reactivated.",
        "Northlake General Hospital",
        False,
        False,
        False,
        hand_authored=True,
    ),
    LintClaim(
        "L59",
        "F70",
        "Anchor Bay Surgical Partners may take a meeting with Brightline Medical after all.",
        "Meadowbrook Community Hospital",
        False,
        False,
        False,
        hand_authored=True,
    ),
    LintClaim(
        "L60",
        "F23",
        "Northlake General Hospital's alarm complaints could resurface next quarter.",
        "Larkspur Memorial Hospital",
        False,
        False,
        False,
        hand_authored=True,
    ),
]

assert len(LINT_CLAIMS) == 60, len(LINT_CLAIMS)
