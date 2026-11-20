"""A tiny hand-assembled v2 world directory for the WP9 bench tests, plus a
scripted extractor output and scripted judges with known expected scores.

Three sources: a call (three labels, one of them silver, one decoy, one
real but unlabeled praise line), a CRM account row (two field labels) and
one email message (one label). Every span is computed with `str.index` on
the exact prepared text, as WP3/WP4 compute theirs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from portolan.bench.disputes import JudgeReply
from portolan.extract.schema import Coverage, ExtractionResult, Fact

CALL_ID = "S-call-w1"
CRM_ID = "S-crm-A4001"
EMAIL_ID = "msg-1@mail.vendor.test"

CALL_TEXT = (
    "Quarterly review, Marrow General\n"
    "[00:00:05] Dana Reyes: Morning all.\n"
    "[00:00:12] Dana Reyes: The Tidewater cartridges arrived four days late again last week.\n"
    "[00:00:30] Sam Ortiz: We'll ship the replacement pumps by Friday, I promise.\n"
    "[00:00:45] Dana Reyes: If Corvane ever undercut you we might look, but we are not "
    "looking.\n"
    "[00:01:02] Dana Reyes: Honestly the new dashboard saves my nurses an hour a day.\n"
    "[00:01:20] Sam Ortiz: Our renewal quote is 42,000 dollars for next year."
)
CRM_TEXT = (
    "[crm/accounts.csv#A-4001] account: crm_id=A-4001, name=Marrow General, arr=120000, "
    "renewal_date=2027-01-31, competitors=Corvane"
)
EMAIL_TEXT = (
    "[msg-1@mail.vendor.test#p1] Hi Sam, thanks for the call.\n"
    "[msg-1@mail.vendor.test#p2] Can you send the updated invoice by Tuesday?"
)

F1 = "The Tidewater cartridges arrived four days late again last week."
F2 = "We'll ship the replacement pumps by Friday, I promise."
F3 = "Our renewal quote is 42,000 dollars for next year."
F3_RESTATED = "42,000 dollars"
DECOY = "If Corvane ever undercut you we might look, but we are not looking."
PRAISE = "Honestly the new dashboard saves my nurses an hour a day."
F5 = "Can you send the updated invoice by Tuesday?"


def _span(text: str, needle: str) -> tuple[int, int]:
    start = text.index(needle)
    return start, start + len(needle)


def _label(
    fact_id: str, fact_type: str, doc_id: str, text: str, needle: str, **extra: Any
) -> dict[str, Any]:
    start, end = _span(text, needle)
    return {
        "fact_id": fact_id,
        "fact_type": fact_type,
        "doc_id": doc_id,
        "source_type": extra.pop("source_type", "call"),
        "locator": extra.pop("locator", "00:00:12"),
        "char_start": start,
        "char_end": end,
        "text": needle,
        "account_id": "A-marrow",
        "date": "2026-05-04",
        **extra,
    }


def labels() -> list[dict[str, Any]]:
    return [
        _label(
            "F1",
            "complaint",
            CALL_ID,
            CALL_TEXT,
            F1,
            label_source="scripted_writer",
            plan_id="D0001",
            slot_id="s1",
            writer_model="qwen/qwen3-32b",
            noise=["asr"],
            asr_in_span=True,
        ),
        _label(
            "F2",
            "commitment",
            CALL_ID,
            CALL_TEXT,
            F2,
            label_source="scripted_writer",
            locator="00:00:30",
            directive="hedged",
            reference_mode="pronoun",
        ),
        _label(
            "D0001-C1",
            "pricing",
            CALL_ID,
            CALL_TEXT,
            F3,
            label_source="silver_completeness",
            locator="00:01:20",
            teacher_model="moonshotai/kimi-k2",
            claim="Marrow General was quoted 42,000 dollars for its renewal.",
        ),
        # A WP5 restatement: a second silver locator for the same fact id.
        _label(
            "D0001-C1",
            "pricing",
            CALL_ID,
            CALL_TEXT,
            F3_RESTATED,
            label_source="silver_completeness",
            locator="00:01:20",
            restates_fact_id="D0001-C1",
        ),
        _label(
            "CF-A-marrow-arr-01",
            "pricing",
            CRM_ID,
            CRM_TEXT,
            "120000",
            source_type="crm",
            locator="crm/accounts.csv#A-4001",
        ),
        _label(
            "CF-A-marrow-competitors-02",
            "competitor_mention",
            CRM_ID,
            CRM_TEXT,
            "Corvane",
            source_type="crm",
            locator="crm/accounts.csv#A-4001",
        ),
        _label(
            "F5",
            "request",
            EMAIL_ID,
            EMAIL_TEXT,
            F5,
            label_source="scripted_writer",
            source_type="email",
            locator="msg-1@mail.vendor.test#p2",
            noise=["forwarded_or_quoted"],
            copy_of=None,
        ),
        # Excluded by the reader: text does not match the span.
        {**_label("F9", "risk", CALL_ID, CALL_TEXT, F1), "text": "something else"},
        # Excluded by the reader: no such source.
        _label("F10", "risk", CALL_ID, CALL_TEXT, F1) | {"doc_id": "S-call-missing"},
    ]


def decoys() -> list[dict[str, Any]]:
    start, end = _span(CALL_TEXT, DECOY)
    return [
        {
            "decoy_id": "DC1",
            "decoy_class_id": "hypothetical",
            "near_type": "competitor_mention",
            "plan_id": "D0001",
            "slot_id": "d1",
            "doc_id": CALL_ID,
            "source_type": "call",
            "locator": "00:00:45",
            "prepared_locator": "00:00:45",
            "char_start": start,
            "char_end": end,
            "text": DECOY,
            "account_id": "A-marrow",
            "writer_model": "qwen/qwen3-32b",
        }
    ]


def sources() -> list[dict[str, Any]]:
    return [
        {
            "source_id": CALL_ID,
            "source_type": "call",
            "kind": "call",
            "date": "2026-05-04",
            "account_id": "A-marrow",
            "participants": ["Dana Reyes", "Sam Ortiz"],
            "noise": [{"type": "asr", "params": {"wer": 0.02}}],
            "text": CALL_TEXT,
        },
        {
            "source_id": CRM_ID,
            "source_type": "crm",
            "kind": "crm_account",
            "date": "2026-10-31",
            "account_id": "A-marrow",
            "text": CRM_TEXT,
        },
        {
            "source_id": EMAIL_ID,
            "source_type": "email",
            "kind": "email",
            "date": "2026-05-06",
            "account_id": "A-marrow",
            "noise": ["forwarded_or_quoted"],
            "text": EMAIL_TEXT,
        },
    ]


WORLD_YAML = """\
id: world-test-7
vendor:
  name: Halcyon Diagnostics
  products:
    - {name: Tidewater, aliases: [TW]}
  competitors:
    - {name: Corvane, aliases: []}
accounts:
  - {id: A-marrow, name: Marrow General, aliases: [Marrow]}
facts:
  - {id: F1, directive: colloquial, reference_mode: explicit, claim: "Marrow General reported \
Tidewater cartridges arriving four days late."}
  - {id: F2, directive: passive, reference_mode: explicit, claim: "Sam Ortiz committed to ship \
replacement pumps by Friday."}
  - {id: F5, directive: embedded, reference_mode: first_name, claim: "Marrow General asked for \
the updated invoice by Tuesday."}
"""


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build_world(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "sources.jsonl", sources())
    _write_jsonl(root / "labels.jsonl", labels())
    _write_jsonl(root / "decoys.jsonl", decoys())
    (root / "world.yaml").write_text(WORLD_YAML, encoding="utf-8")
    return root


def _fact(fid: str, ftype: str, claim: str, quote: str | None, locator: str) -> Fact:
    return Fact(
        id=fid,
        type=ftype,
        claim=claim,
        entities=["A-marrow"],
        locator=locator,
        quote=quote,
        sentiment=0,
        confidence=0.8,
    )


def _result(source_id: str, facts: list[Fact]) -> ExtractionResult:
    return ExtractionResult(
        source_id=source_id,
        summary="scripted",
        facts=facts,
        coverage=Coverage(segments_read=1, segments_total=1),
    )


def scripted_results() -> dict[str, ExtractionResult]:
    """Eight produced facts with one known category each:

    call: f1 matched (exact quote), f2 matched (curly apostrophe: normalized
    tier), f3 decoy hit, f4 off-span but real (the praise line), f5 off-span
    invented (quote not in the text); CRM: f1 type confusion (`question` on
    the `pricing` arr field), f2 matched by containment; email: f1 matched
    through its locator (no quote)."""
    return {
        CALL_ID: _result(
            CALL_ID,
            [
                _fact("f1", "complaint", "Cartridges arrived four days late.", F1, "00:00:12"),
                _fact(
                    "f2",
                    "commitment",
                    "Sam will ship replacement pumps by Friday.",
                    "We’ll ship the replacement pumps by Friday",
                    "00:00:30",
                ),
                _fact(
                    "f3",
                    "competitor_mention",
                    "Marrow General is weighing Corvane.",
                    "If Corvane ever undercut you we might look",
                    "00:00:45",
                ),
                _fact(
                    "f4",
                    "praise",
                    "The new dashboard saves Marrow General nurses an hour a day.",
                    "the new dashboard saves my nurses an hour a day",
                    "00:01:02",
                ),
                _fact(
                    "f5",
                    "risk",
                    "Marrow General plans to move everything to Corvane.",
                    "We are moving everything to Corvane next month",
                    "00:01:20",
                ),
            ],
        ),
        CRM_ID: _result(
            CRM_ID,
            [
                _fact(
                    "f1", "question", "What is the ARR?", "arr=120000", "crm/accounts.csv#A-4001"
                ),
                _fact(
                    "f2",
                    "competitor_mention",
                    "Corvane is a competitor at Marrow General.",
                    "competitors=Corvane",
                    "crm/accounts.csv#A-4001",
                ),
            ],
        ),
        EMAIL_ID: _result(
            EMAIL_ID,
            [
                _fact(
                    "f1",
                    "request",
                    "Marrow General asked for the updated invoice by Tuesday.",
                    None,
                    "msg-1@mail.vendor.test#p2",
                )
            ],
        ),
    }


def verdict(v: str, evidence: str = "", type_fits: bool = True) -> dict[str, Any]:
    return {"verdict": v, "type_fits": type_fits, "evidence": evidence, "rationale": f"{v}."}


class ScriptedJudges:
    """`JudgeBackend` double: `script[(model, needle)]` is the reply for the
    first dispute whose prompt contains `needle`; anything else is
    `unsupported`. Records every call."""

    def __init__(self, script: dict[tuple[str, str], dict[str, Any]], cost: float = 0.001):
        self.script = script
        self.cost = cost
        self.calls: list[tuple[str, str]] = []

    def judge(self, *, model: str, system: str, user: str, max_tokens: int) -> JudgeReply:
        self.calls.append((model, user))
        for (m, needle), data in self.script.items():
            if m == model and needle in user:
                return JudgeReply(data=data, cost_usd=self.cost, input_tokens=900, output_tokens=40)
        return JudgeReply(
            data=verdict("unsupported"), cost_usd=self.cost, input_tokens=900, output_tokens=40
        )


JUDGES = ("judge/a", "judge/b", "judge/c")
PRICES = dict.fromkeys(JUDGES, (1e-7, 4e-7))


def world_script() -> dict[tuple[str, str], dict[str, Any]]:
    """The praise line: 2-1 supported (judge b dissents; a and c quote the
    document). The decoy and the invented risk: a and b say unsupported (the
    default), so c is never called for them."""
    return {
        ("judge/a", "Claim: The new dashboard"): verdict("supported", PRAISE),
        ("judge/b", "Claim: The new dashboard"): verdict("unsupported"),
        ("judge/c", "Claim: The new dashboard"): verdict(
            "supported", "the new dashboard saves my nurses"
        ),
    }
