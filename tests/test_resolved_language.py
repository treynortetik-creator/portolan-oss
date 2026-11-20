"""`portolan.report.resolved_language`: the sentences the 2026-09-20 live
review queue rejected although they assert no resolution, and the attack
sentences from the earlier reviews that must still be rejected.

The kept sentences are taken from that queue (backup CSV of 2026-09-20
06:22 MST, rows 169 to 221, trimmed to the clause that tripped the old rule),
checked against the themes that week's report carried."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from portolan.registry.text import slugify
from portolan.report.resolved_language import resolved_language_rejected, themes_named_in


@dataclass
class _Theme:
    name: str
    movement: str

    @property
    def slug(self) -> str:
        return f"theme-{slugify(self.name)}"


W36_THEMES = [
    _Theme("Brightline Medical Competitive Push", "resolved"),
    _Theme("Healthcare Renewals Face Discount Pressure", "resolved"),
    _Theme("Quick Turnaround Praise", "resolved"),
    _Theme("Tidewater T2 Delivery Delays", "resolution proposed"),
    _Theme("Larkspur Biomed Engineering Praise", "rising"),
    _Theme("Tidewater Expansion And Halvardcare Renewals", "falling"),
    _Theme("Follow-up Commitments Piling Up", "falling"),
]

CLAIMS = [
    "June Park said the August delivery issues seem to be behind us.",
    "The delivery issues from August seem to be behind us, June Park wrote.",
]

KEPT = [
    # A need, not a state (rows 171, 175, 176, 188).
    "Halvard Support committed to follow up with delivery details, making the delivery track "
    "record something to resolve before that pilot begins.",
    "A third repeated ask for the same integration signals growing frustration that the "
    "promised timeline needs to resolve.",
    "The account has had two late Tidewater T2 deliveries in August that are worth resolving "
    "before the pilot begins.",
    "Push Sam Okafor's logistics working session for Northlake Health System to resolve the "
    "pattern behind the two-day-late Tidewater T3 delivery. ^[S-call-011 @ 00:02:46]",
    # Negated (rows 187, 191, 215, 217).
    "Four actions remain open with none yet overdue and none resolved this week.",
    "None of these threads shows a resolution yet this week, and the broader Tidewater T2 "
    "Delivery Delays theme is marked rising across nine accounts.",
    "Leaving a direct customer question unanswered risks becoming a satisfaction issue if it "
    "isn't closed out soon.",
    "The account also has an open request for a small-volume consumables discount that "
    "Halvard has not yet closed out.",
    # The customer's own words, quoted and cited (row 180, repeated in 183, 203, 206, 214,
    # 216, 220, 222).
    "Meadowbrook Community Hospital's pharmacy director said on September 4, 2026 that the "
    "August delivery issues 'seem to be behind us' ^[S-email-072 @ p3].",
    # The proposed state, named on the proposed theme (rows 202, 205, 219, 221).
    "Even so, the portfolio-wide Tidewater T2 Delivery Delays theme is only marked "
    "'resolution proposed,' not confirmed resolved.",
    "That said, Tidewater T2 Delivery Delays is a different story: its status is only "
    "resolution proposed, meaning a judge flagged it and a human still needs to confirm.",
    "The hospital also decided to drop the idea of bringing in a second supplier for now, "
    "though the broader Tidewater T2 Delivery Delays theme still shows only a proposed "
    "resolution, not a confirmed one, across accounts.",
    # Resolved themes named in full; "and", and the "praise" inside "Quick Turnaround
    # Praise", no longer name another theme (row 169's shape).
    "The biggest news is a cluster of resolutions: Healthcare Renewals Face Discount "
    "Pressure, Quick Turnaround Praise, and Brightline Medical Competitive Push all closed "
    "out this week.",
]

REJECTED = [
    # Assertions about the proposed theme, paraphrased or not.
    "Tidewater T2 Delivery Delays is now resolved.",
    "Late deliveries will resolve next week.",
    "The delivery delays are behind us now.",
    "Meadowbrook's pharmacy director already considers the delivery issues resolved.",
    "Tidewater T2 Delivery Delays sits at resolution proposed, and the delays are fixed.",
    # A quote that is not a recorded claim, too short to be one, or not cited.
    "The delivery issues 'are finally all behind us now' ^[S-email-072 @ p3].",
    "The delivery issues are 'behind us' ^[S-email-072 @ p3].",
    "The August delivery issues 'seem to be behind us'.",
    # A resolved theme named alongside a theme that is not resolved.
    "Quick Turnaround Praise and Larkspur Biomed Engineering Praise both resolved.",
    "Healthcare Renewals Face Discount Pressure resolved, and so did Tidewater expansion.",
    # A need-form word next to a real assertion.
    "The delays needed to resolve, and they are resolved.",
]


@pytest.mark.parametrize("sentence", KEPT)
def test_sentences_that_assert_no_resolution_are_kept(sentence: str) -> None:
    assert not resolved_language_rejected(sentence, W36_THEMES, grounded_texts=CLAIMS)


@pytest.mark.parametrize("sentence", REJECTED)
def test_sentences_that_assert_an_unbacked_resolution_are_rejected(sentence: str) -> None:
    assert resolved_language_rejected(sentence, W36_THEMES, grounded_texts=CLAIMS)


def test_proposed_resolution_wording_is_rejected_when_no_named_theme_is_proposed() -> None:
    themes = [_Theme("Quick Turnaround Praise", "rising"), _Theme("Late deliveries", "falling")]
    assert resolved_language_rejected("Quick Turnaround Praise has a proposed resolution.", themes)


def test_every_resolution_word_is_rejected_on_a_week_with_no_resolved_theme() -> None:
    themes = [_Theme("Late deliveries", "resolution proposed")]
    for sentence in (
        "Late deliveries is now resolved.",
        "Late deliveries will resolve next week.",
        "Late deliveries is resolving at two accounts.",
        "The late deliveries resolution is final.",
        "Meadowbrook's late-delivery complaints are fixed.",
        "The late-delivery issue has been cleared up.",
        "Late deliveries are behind us now.",
    ):
        assert resolved_language_rejected(sentence, themes), sentence


def test_a_word_inside_another_named_theme_does_not_name_a_theme() -> None:
    sentence = "Quick Turnaround Praise resolved at Northlake and St. Aldric."
    assert themes_named_in(sentence, W36_THEMES) == {"theme-quick-turnaround-praise"}
