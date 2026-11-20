"""The overlap gate's numeral and name rules against real false drops and
against invented content that must still drop.

Each false-drop case is a real fact from `var/data/silver/<corpus>/
dropped.jsonl` (2026-09-22 snapshot) with the record text it was checked
against, trimmed to the lines that matter. No network, no corpus files.
"""

from __future__ import annotations

import pytest

from portolan.training.overlap_gate import (
    overlap_gate_reason,
    unsupported_names,
    unsupported_numerals,
)

SAAS_5K = (
    "[00:04:30] Sales Rep: It’s $5k/month. I know that sounds like a lot, but considering "
    "what it offers, it might save you more long-term."
)
CFPB_5_YEAR = (
    "Consumer: I signed up with Tally services to help lower my credit card debt. "
    "Now, this company is charging me more than before and have me on a 5 year payment "
    "plan. This company passed on my balance to a debt collection company."
)
CFPB_COMERICA = (
    "[S-ticket-cfpb-10148662#1] Consumer: XXXX XXXX Comerica sent my debit card loaded "
    "with funds to somewhere else activated. The money 's gone and they refused to speak "
    "to me"
)
FDA_JUL = (
    "Reporter: It was reported that the patient experienced infusion set was leaking at "
    "the site event on 31 Jul 2026."
)
FDA_SMARTGUARD = (
    "Reporter: It was reported to Minimed that the customer experienced Hyperglycemia "
    "with a possible under delivery anomaly as the smart guard did not give bolus. The "
    "customer reported blood glucose value of 366 mg/dL"
)
ENRON_30M = (
    "Northern Gas Pipeline. The ARC consortium needs to fund approximately $30M to develop "
    "this project."
)
ENRON_FIAT = (
    "Montedison SpA (I.MNT) is preparing its defense of a Fiat SpA (FIA)-led takeover bid. "
    "A U.S. industrial partner, such as AEP (AEPI), Enron (ENE) could also aid in a defense "
    "strategy"
)

REAL_FALSE_DROPS = [
    pytest.param(
        "SyncERP is priced at $5,000 per month", SAAS_5K, id="saas-13-conv-531-5k-per-month"
    ),
    pytest.param(
        "The new debt company is charging more than Tally did and placed the consumer on a "
        "5-year payment plan",
        CFPB_5_YEAR,
        id="cfpb-10311867-5-year",
    ),
    pytest.param(
        "Comerica Bank sent the consumer's debit card loaded with funds to an incorrect "
        "location where it was activated by an unauthorized party, resulting in stolen funds, "
        "and the bank refuses to speak with the consumer.",
        CFPB_COMERICA,
        id="cfpb-10148662-comerica-bank",
    ),
    pytest.param(
        "Comenity Bank incorrectly reported a 30-day late payment and $22 late fee on a "
        "previously paid-off Buckle credit card",
        "I have a Buckle account and the credit card for that account is with Comenity Bank, "
        "This credit card was previously paid off and someone called me and told me I was 30 "
        "days late for a {$22.00} payment and that was a {$22.00} late fee",
        id="cfpb-10521325-30-day",
    ),
    pytest.param(
        "Consumer's Direct Express card repeatedly fails at ATMs despite company assurances "
        "it would work.",
        "Consumer: I've tried several times to use an ATM to withdraw my money but my card "
        "never works. I contacted direct express multiple times to resolve this issue.",
        id="cfpb-10392385-atms",
    ),
    pytest.param(
        "A patient experienced an infusion set leaking at the site on 2026-07-31",
        FDA_JUL,
        id="fda-50516779-iso-date",
    ),
    pytest.param(
        "BD Pyxis MedStation ES Auxiliary System serial number manufactured 2021-05-08 had no "
        "prior similar complaints",
        "BD Pyxis MedStation ES Auxiliary System. A review of the device history record for SN "
        "(b)(6)was performed from the date of manufacture, 08-MAY-2021 and confirmed that "
        "this device was not previously returned",
        id="fda-50516400-dd-mon-yyyy",
    ),
    pytest.param(
        "Customer experienced hyperglycemia with possible under-delivery anomaly as "
        "SmartGuard did not give bolus, with blood glucose of 366 mg/dL",
        FDA_SMARTGUARD,
        id="fda-50516562-smartguard",
    ),
    pytest.param(
        "ARC consortium needs approximately $30 million to develop the Northern Gas Pipeline "
        "project",
        ENRON_30M,
        id="enron-089311-30m",
    ),
    pytest.param(
        "Jana Giovannini requested David Delainey provide a recommendation for a Director by "
        "October 13, 2000",
        "From: Jana Giovannini 10/12/2000 02:44 PM To: David W Delainey/HOU/ECT@ECT Dave, I "
        "am following up on the voicemail I left you on Thursday October 5th, requesting a "
        "recommendation for a Director to represent Analysts at the final 2000 PRC. We would "
        "like to finalize this by Friday, October 13th in order to load the PEP system.",
        id="enron-089319-ordinal",
    ),
    pytest.param(
        "Dave Delainey scheduled a strategy meeting for October 17, 2000, from 3:00 PM to "
        "5:00 PM in room 30C1",
        "david.delainey@enron.com REMINDER -----TODAY 3:00 PM Dave would like to have a "
        "strategy meeting next Tuesday  October 17, 2000, from 3:00-5:00 pm in room 30C1.",
        id="enron-089307-time-range",
    ),
    pytest.param(
        "Gopinath Munde demands the judicial inquiry terms be finalized within 8 days and the "
        "final report submitted within 3 months",
        "Gopinath Munde Sunday demanded that the terms of reference for the judicial inquiry "
        "should be finalised within eight days instead of one month. Demanding that the final "
        "report of the judicial probe be submitted within three months",
        id="enron-071512-spelled-numbers",
    ),
    pytest.param(
        "Montedison SpA considering Enron as potential U.S. industrial partner for takeover "
        "defense against Fiat-led bid",
        ENRON_FIAT,
        id="enron-071512-hyphen-compound",
    ),
    pytest.param(
        "David Delainey raised a significant control issue regarding VPs having signing authority",
        "Janet, I completely disagree with legals position. I have difficulty with some of our "
        "VP's having signing authority. I believe this is a significant control issue. Regards "
        "David Delainey",
        id="enron-089338-plural-acronym",
    ),
    pytest.param(
        "effgeee has X session using 25-30% CPU due to something in home directory",
        "[L27] effgeee: grr, something in my home directory is causing my xsession to "
        "constantly use 25 -30% cpu, a new profile does no such thing. X",
        id="irc-w013-range",
    ),
    pytest.param(
        "cdoty asked for recommendation on best Active Directory authentication method for "
        "Ubuntu systems",
        "cdoty: A member of the board requires that all systems use authentication from our "
        "main windows system. does anyone have a recomendation for what activedirectory way "
        "works the best. Ubuntu",
        id="irc-w025-joined-name",
    ),
]


@pytest.mark.parametrize(("claim", "text"), REAL_FALSE_DROPS)
def test_real_false_drops_now_pass(claim: str, text: str) -> None:
    assert unsupported_numerals(claim, text) == []
    assert unsupported_names(claim, text) == []


def test_real_false_drop_passes_the_whole_gate() -> None:
    assert (
        overlap_gate_reason(
            claim="SyncERP is priced at $5,000 per month",
            quote="It’s $5k/month.",
            doc_text=SAAS_5K,
            threshold=0.0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("claim", "text", "missing"),
    [
        ("SyncERP is priced at $5,500 per month", SAAS_5K, ["5,500"]),
        ("SyncERP is priced at $5 per month", SAAS_5K, ["5"]),
        ("SyncERP is priced at $50k per month", SAAS_5K, ["50k"]),
        ("The consumer was placed on a 5-month payment plan", CFPB_5_YEAR, ["5-month"]),
        ("The consumer was placed on a 6-year payment plan", CFPB_5_YEAR, ["6-year"]),
        ("ARC consortium needs approximately $30 billion", ENRON_30M, ["30 billion"]),
        ("ARC consortium needs approximately $300 million", ENRON_30M, ["300 million"]),
        ("ARC consortium needs approximately $3M", ENRON_30M, ["3M"]),
        ("A patient's infusion set leaked on 2026-07-13", FDA_JUL, ["2026-07-13"]),
        ("A patient's infusion set leaked on 2025-07-31", FDA_JUL, ["2025-07-31"]),
        ("An infusion set leaked at 41 sites", FDA_JUL, ["41"]),
    ],
)
def test_invented_numerals_still_drop(claim: str, text: str, missing: list[str]) -> None:
    assert unsupported_numerals(claim, text) == missing


@pytest.mark.parametrize(
    ("claim", "text", "missing"),
    [
        ("Chase Bank sent the consumer's card elsewhere", CFPB_COMERICA, ["Bank"]),
        (
            "The consumer said Wells Fargo Bank sent the card",
            CFPB_COMERICA,
            ["Wells", "Fargo", "Bank"],
        ),
        (
            "The consumer said Comerica and Kenji Mori sent the card",
            CFPB_COMERICA,
            ["Kenji", "Mori"],
        ),
        ("Montedison SpA faces a Renault-led takeover bid", ENRON_FIAT, ["Renault"]),
        ("The SmartShield did not give a bolus", FDA_SMARTGUARD, ["SmartShield"]),
        (
            "The consumer's card fails at ATMs run by Cardtronics",
            CFPB_COMERICA,
            ["ATMs", "Cardtronics"],
        ),
    ],
)
def test_invented_names_still_drop(claim: str, text: str, missing: list[str]) -> None:
    assert unsupported_names(claim, text) == missing


def test_invented_numeral_drops_through_the_whole_gate() -> None:
    assert (
        overlap_gate_reason(
            claim="SyncERP is priced at $5,500 per month",
            quote="It’s $5k/month.",
            doc_text=SAAS_5K,
            threshold=0.0,
        )
        == "overlap_gate_numeral"
    )


def test_identifier_with_digits_stays_whole() -> None:
    text = "Enron decided to bill FTS-2 reservation invoices on June 1 for May production."
    assert unsupported_names("Enron bills FTS-2 reservation invoices", text) == []
    assert unsupported_names("Enron bills FTS-9 reservation invoices", text) == ["FTS-9"]


def test_number_glued_to_letters_is_part_of_a_name() -> None:
    text = "Revenue rose in Q4 and the B2B team grew."
    assert unsupported_numerals("Revenue rose in Q4 for the B2B team", text) == []
    # the text's "Q4" does not offer a free-standing 4 either
    assert unsupported_numerals("Revenue rose for 4 quarters", text) == ["4"]


def test_currency_code_glued_number_is_still_a_numeral() -> None:
    text = "prices soared to EUR1,200 a megawatt-hour from around EUR35.00/MWh"
    assert unsupported_numerals("prices reached EUR1,200 per MWh from EUR35.00", text) == []
    assert unsupported_numerals("prices reached EUR1,900 per MWh", text) == ["1,900"]


def test_claim_scale_word_needs_the_scaled_value() -> None:
    assert unsupported_numerals("About 5 million users logged in", "About 5 users logged in") == [
        "5 million"
    ]
    assert unsupported_numerals("About 5 users logged in", "About 5 million users logged in") == []


def test_grouping_comma_needs_three_digits() -> None:
    text = "Invoice Date of February 28,2001 and a due date 10 calendar days later"
    assert unsupported_numerals("The invoice date was February 28, 2001", text) == []


def test_spaced_scale_suffix_after_currency() -> None:
    # a kept fact (S-email-enron-089321) that must stay kept under value comparison
    text = (
        "Subject: Fee to Bear, Stearns & Co. A bit of good news: Of the $2.582 MM payable to "
        "Bear, Stearns in conjunction with their advisory work"
    )
    assert unsupported_numerals("Enron owes Bear, Stearns & Co. $2.582 million", text) == []
    assert unsupported_numerals("Enron owes Bear, Stearns & Co. $2.582", text) == ["2.582"]


def test_a_digit_run_longer_than_decimal_precision_keeps_every_digit() -> None:
    # IRC windows carry long ids; one crashed the first repair run
    text = "paste id 12345678901234567890123456789012345 and 5.0 GB free"
    assert unsupported_numerals("id 12345678901234567890123456789012345 has 5 GB", text) == []
    assert unsupported_numerals("id 12345678901234567890123456789012346", text) == [
        "12345678901234567890123456789012346"
    ]
