"""Multi-hop questions over the demo graph, for the agentic retrieval bench
(`portolan.bench.agentic`). Evaluation only: never training data.

The graph is `portolan.bench.demo_search.seed_demo_graph` (the synthetic
Halvard corpus, `corpus/out`, through the fake extractor). Each question
was written on 2026-09-23 by reading that graph's claims and links, and
needs facts from two or more notes, or a hop from the note search lands on
(an account's parent, a hospital's system, a product's or competitor's
incoming signals, a contact's replacement). `expected` names the notes
holding the answer; `answer_facts` feed ERB's completeness judge.

`LIVE_CHAT_Q7` is question 7 of the live chat log (2026-09-23): the live
single-shot chat answered "nothing recorded" although the claims exist.
"""

from __future__ import annotations

from portolan.bench.agentic import BenchQuestion

__all__ = ["DEMO_MULTIHOP_QUESTIONS", "LIVE_CHAT_Q7", "demo_question_set"]


def _q(qid: str, question: str, gold: str, facts: tuple[str, ...], expected: tuple[str, ...]):
    return BenchQuestion(qid, question, gold, facts, expected, kind="multi_hop")


LIVE_CHAT_Q7 = BenchQuestion(
    "live_q7",
    "Which accounts have asked for Tidewater Link to work with Epic?",
    "Four accounts: St. Aldric Health (asked on 2026-08-12 and followed up on 2026-08-21), "
    "Driftwood Community Hospital (asked on 2026-08-27), Wrenfield Health (deferred on "
    "2026-08-10, then on 2026-08-24 asked Halvard to begin immediately) and Larkspur Regional "
    "Health (asked for the integration by 2027-01).",
    (
        "St. Aldric Health asked for Tidewater Link integration with its Epic instance.",
        "Driftwood Community Hospital asked for a Tidewater Link connection to its Epic instance.",
        "Wrenfield Health asked Halvard to begin Tidewater Link integration with its Epic "
        "instance.",
        "Larkspur Regional Health asked for Tidewater Link integration with its Epic instance.",
    ),
    (
        "signal-request-st-aldric-health",
        "signal-request-driftwood-community-hospital",
        "signal-request-wrenfield-health",
        "signal-request-larkspur-regional-health",
    ),
    kind="must_answer",
)


DEMO_MULTIHOP_QUESTIONS: tuple[BenchQuestion, ...] = (
    _q(
        "mh_01",
        "Which Larkspur Regional Health hospitals reported late Tidewater T2 deliveries in "
        "August 2026?",
        "Larkspur Memorial Hospital (a T2 consumable delivery four days late on 2026-08-04) and "
        "Larkspur North Hospital (an order placed 2026-08-08 arrived three days late).",
        (
            "Larkspur Memorial Hospital had a Tidewater T2 delivery arrive late in August 2026.",
            "Larkspur North Hospital had a Tidewater T2 order arrive late in August 2026.",
        ),
        ("signal-complaint-larkspur-memorial-hospital", "signal-complaint-larkspur-north-hospital"),
    ),
    _q(
        "mh_02",
        "Who owns the health system that Larkspur Memorial Hospital belongs to?",
        "Larkspur Memorial Hospital is part of Larkspur Regional Health, which is owned by "
        "Pinecrest Health Partners.",
        (
            "Larkspur Memorial Hospital is part of Larkspur Regional Health.",
            "Larkspur Regional Health is owned by Pinecrest Health Partners.",
        ),
        ("larkspur-memorial-hospital", "signal-other-larkspur-regional-health"),
    ),
    _q(
        "mh_03",
        "What is the annual recurring revenue of the system that owns Wrenfield Medical Center, "
        "and how many pumps does Wrenfield Medical Center run?",
        "Wrenfield Medical Center is owned by Wrenfield Health, whose annual recurring revenue "
        "is $1,410,000 (as of 2026-09-11); the medical center runs eighteen Tidewater T2 pumps "
        "across its ICU and med-surg floors.",
        (
            "Wrenfield Medical Center is owned by Wrenfield Health.",
            "Wrenfield Health's annual recurring revenue is $1,410,000.",
            "Wrenfield Medical Center runs eighteen Tidewater T2 pumps.",
        ),
        (
            "signal-other-wrenfield-medical-center",
            "signal-other-wrenfield-health",
            "signal-usage-wrenfield-medical-center",
        ),
    ),
    _q(
        "mh_04",
        "Ashgrove Health's CFO warned that late deliveries would come up at renewal. When is "
        "that renewal now, and what was the date before?",
        "The HalvardCare renewal was 2027-03-31 and was moved to 2027-06-30 at Ashgrove's "
        "request on 2026-08-20, the same day the CFO said continued late deliveries would be "
        "raised at the 2027 renewal.",
        (
            "Ashgrove Health's CFO said continued late deliveries would be raised at the 2027 "
            "renewal.",
            "The HalvardCare renewal is now 2027-06-30.",
            "The renewal was previously 2027-03-31.",
        ),
        ("signal-risk-ashgrove-health", "signal-decision-ashgrove-health"),
    ),
    _q(
        "mh_05",
        "Which Ashgrove Health hospitals had late Tidewater T2 deliveries, and did deliveries "
        "improve afterwards?",
        "Ashgrove Medical Center had two late deliveries in the week of 2026-07-27 and Ashgrove "
        "Riverside Hospital's delivery arrived five days late on 2026-08-18. Deliveries "
        "improved: on 2026-08-31 Ashgrove Medical Center's ICU nurse manager said the last two "
        "deliveries came in right on schedule.",
        (
            "Ashgrove Medical Center had late Tidewater T2 deliveries.",
            "Ashgrove Riverside Hospital had a Tidewater T2 delivery arrive five days late.",
            "By 2026-08-31 Ashgrove Medical Center said recent deliveries were on schedule.",
        ),
        (
            "signal-complaint-ashgrove-medical-center",
            "signal-complaint-ashgrove-riverside-hospital",
            "signal-praise-ashgrove-medical-center",
        ),
    ),
    _q(
        "mh_06",
        "What did Halvard promise Ashgrove Health about the delivery problems, who made the "
        "promise, and what operational fix followed?",
        "Sam Okafor promised Ashgrove Health a written delivery root-cause summary by "
        "2026-08-28; Halvard's operations manager later confirmed that the Dayton distribution "
        "branch hired two additional drivers, effective 2026-08-25.",
        (
            "Sam Okafor made the promise.",
            "The promise was a written delivery root-cause summary by 2026-08-28.",
            "The Dayton distribution branch hired two additional drivers.",
        ),
        ("signal-commitment-ashgrove-health", "signal-decision-ashgrove-medical-center"),
    ),
    _q(
        "mh_07",
        "Northlake Health System said in June 2026 that it was not evaluating other pump "
        "vendors. What changed, and what delivery problems was its hospital having?",
        "In August 2026 Northlake began evaluating Corvane Systems pumps for Northlake "
        "General's new tower. Northlake General Hospital had late Tidewater T3 deliveries: two "
        "days late on 2026-08-11, three days late on 2026-08-13, and an order of 2026-08-15 "
        "three days late, the third late delivery in three weeks.",
        (
            "Northlake began evaluating Corvane Systems pumps in August 2026.",
            "The evaluation was for Northlake General's new tower.",
            "Northlake General Hospital had several late Tidewater T3 deliveries in August 2026.",
        ),
        (
            "signal-competitor_mention-northlake-health-system",
            "signal-complaint-northlake-general-hospital",
        ),
    ),
    _q(
        "mh_08",
        "Which competitor's pumps is Northlake evaluating, and what happened to that "
        "competitor's pump line afterwards?",
        "Corvane Systems; on 2026-08-26 Corvane issued a voluntary recall of its C-40 pump line.",
        (
            "Northlake is evaluating Corvane Systems pumps.",
            "Corvane Systems issued a voluntary recall of its C-40 pump line on 2026-08-26.",
        ),
        (
            "signal-competitor_mention-northlake-health-system",
            "signal-competitor_mention-corvane-systems",
        ),
    ),
    _q(
        "mh_09",
        "Which accounts have been in contact with Brightline Medical, and what came of each?",
        "Kestrel Medical Group left Halvard for Brightline in 2026-02; Vireo Surgical Group "
        "received a Brightline proposal in 2026-05 it considered 15 percent cheaper; a "
        "Brightline rep reached out to Anchor Bay Surgical Partners, which had not taken a "
        "meeting (2026-08-15).",
        (
            "Kestrel Medical Group moved to Brightline Medical.",
            "Vireo Surgical Group received a Brightline proposal about 15 percent cheaper.",
            "Anchor Bay Surgical Partners was contacted by a Brightline rep but had not taken a "
            "meeting.",
        ),
        (
            "signal-decision-kestrel-medical-group",
            "signal-competitor_mention-vireo-surgical-group",
            "signal-competitor_mention-anchor-bay-surgical-partners",
        ),
    ),
    _q(
        "mh_10",
        "What commitments has Dana Whitlock made, and to which accounts?",
        "A quarterly business review with Northlake Health System starting 2026-10, and a "
        "formal Tidewater Link project timeline for Driftwood Community Hospital by 2026-09-05.",
        (
            "Dana Whitlock committed to a quarterly business review with Northlake Health System.",
            "Dana Whitlock committed to send Driftwood Community Hospital a formal Tidewater Link "
            "project timeline.",
        ),
        (
            "signal-commitment-northlake-health-system",
            "signal-commitment-driftwood-community-hospital",
        ),
    ),
    _q(
        "mh_11",
        "Driftwood Community Hospital asked for a Tidewater Link connection to Epic. What did "
        "Halvard commit to in response, and has Driftwood looked at competitors?",
        "Dana Whitlock committed to send Driftwood a formal Tidewater Link project timeline by "
        "2026-09-05. Driftwood briefly compared Corvane Systems pumps last year, but it never "
        "went anywhere.",
        (
            "Halvard committed to a formal Tidewater Link project timeline for Driftwood.",
            "The timeline was due by 2026-09-05.",
            "Driftwood briefly compared Corvane Systems pumps, but it never went anywhere.",
        ),
        (
            "signal-request-driftwood-community-hospital",
            "signal-commitment-driftwood-community-hospital",
            "signal-competitor_mention-driftwood-community-hospital",
        ),
    ),
    _q(
        "mh_12",
        "How did Wrenfield Health's position on the Tidewater Link/Epic integration change, and "
        "what did Halvard promise it?",
        "On 2026-08-10 Wrenfield deferred the integration until 2027; on 2026-08-24 it reversed "
        "course and asked Halvard to begin immediately. Priya Natarajan promised a Tidewater "
        "Link/Epic integration test plan by 2026-09-10.",
        (
            "Wrenfield Health first deferred the integration until 2027.",
            "Wrenfield Health then asked Halvard to begin the integration immediately.",
            "Priya Natarajan promised an integration test plan by 2026-09-10.",
        ),
        ("signal-request-wrenfield-health", "signal-commitment-wrenfield-health"),
    ),
    _q(
        "mh_13",
        "Wrenfield Health asked for the Tidewater Link integration to start right away. Did any "
        "of its hospitals ask whether that timeline covered them, and did they get an answer?",
        "Yes: Wrenfield East Hospital asked on 2026-09-05 whether the Tidewater Link "
        "integration timeline applied to its facility too, and did not receive an answer in "
        "the same thread.",
        (
            "Wrenfield East Hospital asked whether the Tidewater Link timeline applied to it.",
            "The question was not answered.",
        ),
        ("signal-question-wrenfield-east-hospital", "signal-request-wrenfield-health"),
    ),
    _q(
        "mh_14",
        "Cobalt Ridge Medical Center expects a credit on its next invoice. Why, and who is the "
        "materials manager behind the request?",
        "Because of late Tidewater T2 deliveries: three days late on 2026-08-06, four days "
        "overdue on 2026-08-11, and four days late on 2026-08-22; it expects a delivery-delay "
        "credit on its next HalvardCare invoice. The materials manager is Aisha Bell, who took "
        "over after Tom Ridley left in 2026-07.",
        (
            "Cobalt Ridge expects a delivery-delay credit on its next HalvardCare invoice.",
            "Cobalt Ridge had several late Tidewater T2 deliveries in August 2026.",
            "Aisha Bell is Cobalt Ridge's materials manager.",
            "Aisha Bell replaced Tom Ridley.",
        ),
        (
            "signal-pricing-cobalt-ridge-medical-center",
            "signal-complaint-cobalt-ridge-medical-center",
            "tom-ridley",
        ),
    ),
    _q(
        "mh_15",
        "Which hospital lent Tidewater T2 consumables to Cobalt Ridge, and what did that "
        "hospital's biomed lead say about switching suppliers?",
        "Larkspur Memorial Hospital. Its biomed lead said on 2026-08-19 that if deliveries did "
        "not stabilize by September he would recommend piloting a second consumables supplier; "
        "on 2026-09-02, after two weeks of on-time deliveries, he dropped the idea for now.",
        (
            "Cobalt Ridge borrowed consumables from Larkspur Memorial Hospital.",
            "Larkspur Memorial's biomed lead threatened to recommend a second consumables "
            "supplier.",
            "He later dropped the second-supplier idea after deliveries were on time.",
        ),
        (
            "signal-complaint-cobalt-ridge-medical-center",
            "signal-risk-larkspur-memorial-hospital",
            "signal-praise-larkspur-memorial-hospital",
        ),
    ),
    _q(
        "mh_16",
        "Larkspur Regional Health plans to standardise on the Tidewater T3. By when, and how "
        "many T2 pumps and T3 trial units does Larkspur Memorial have?",
        "It decided on 2026-06-09 to standardise both hospitals on the Tidewater T3 by 2027-06. "
        "Larkspur Memorial Hospital has forty Tidewater T2 pumps and is trialling four T3 units "
        "(as of 2026-08-05).",
        (
            "Larkspur Regional Health will standardise on the Tidewater T3 by 2027-06.",
            "Larkspur Memorial has forty Tidewater T2 pumps.",
            "Larkspur Memorial is trialling four T3 units.",
        ),
        ("signal-decision-larkspur-regional-health", "signal-usage-larkspur-memorial-hospital"),
    ),
    _q(
        "mh_17",
        "What discount does Larkspur Regional Health expect at its 2027 renewal, and who became "
        "its CFO in September 2026?",
        "A 12 percent volume discount on Tidewater T2 consumables, which its VP Supply Chain "
        'said on 2026-08-25 "isn\'t optional anymore"; Renata Solis became Chief Financial '
        "Officer effective 2026-09-01.",
        (
            "Larkspur Regional Health expects a 12 percent volume discount on Tidewater T2 "
            "consumables.",
            "Renata Solis became Larkspur Regional Health's CFO.",
        ),
        (
            "signal-pricing-larkspur-regional-health",
            "signal-contact_change-larkspur-regional-health",
        ),
    ),
    _q(
        "mh_18",
        "What questions or feature requests have customers raised about the Tidewater T3, and "
        "were any answered?",
        "Northlake Health System asked on 2026-06-17 whether the Tidewater T3 supports Cerner "
        "integration and did not receive an answer; Ashgrove Medical Center's ICU asked for a "
        "night-mode alarm volume on the T3 (no answer recorded).",
        (
            "Northlake Health System asked whether the Tidewater T3 supports Cerner integration.",
            "Northlake's question was not answered.",
            "Ashgrove Medical Center's ICU asked for a night-mode alarm volume on the T3.",
        ),
        ("signal-question-northlake-health-system", "signal-request-ashgrove-medical-center"),
    ),
    _q(
        "mh_19",
        "Meadowbrook Community Hospital's pharmacy director said they were looking at other "
        "pump options. What delivery problems did Meadowbrook have, and were they resolved?",
        "Its Tidewater T2 order of 2026-08-12 arrived five days late and a second delivery was "
        "four days late on 2026-08-19; on 2026-09-04 the pharmacy director said the August "
        'delivery issues "seem to be behind us."',
        (
            "Meadowbrook had a Tidewater T2 order arrive five days late.",
            "Meadowbrook had a second late delivery, four days behind schedule.",
            "By 2026-09-04 the pharmacy director said the delivery issues seemed to be behind "
            "them.",
        ),
        (
            "signal-competitor_mention-meadowbrook-community-hospital",
            "signal-complaint-meadowbrook-community-hospital",
            "signal-praise-meadowbrook-community-hospital",
        ),
    ),
    _q(
        "mh_20",
        "Who replaced Nina Vale at Vireo Surgical Group, and what competitor proposal had Vireo "
        "received while she was there?",
        "Diego Ramos succeeded Nina Vale, who left in 2026-08. In 2026-05 Vireo received a "
        "Brightline Medical proposal it considered 15 percent cheaper than Halvard's.",
        (
            "Diego Ramos succeeded Nina Vale.",
            "Vireo received a Brightline Medical proposal.",
            "The proposal was considered 15 percent cheaper than Halvard's.",
        ),
        ("nina-vale", "signal-competitor_mention-vireo-surgical-group"),
    ),
)


def demo_question_set() -> tuple[BenchQuestion, ...]:
    """The 20 multi-hop questions plus the live chat's question 7."""
    return (*DEMO_MULTIHOP_QUESTIONS, LIVE_CHAT_Q7)
