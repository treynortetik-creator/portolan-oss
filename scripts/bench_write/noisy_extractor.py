"""Write-path stress run: the synthetic corpus through a deliberately sloppy
extractor, then count what happened to every fact.

`--llm fake` emits exactly what the truth table planted, so it cannot show
what the write path does with the mistakes a cheap model makes. This wraps
that fake and damages its output the same way every time (seeded), in the
ways real extraction runs have been seen to fail:

    no_entity         the fact names nobody
    unknown_only      it names only a name the registry does not have
    extra_unknown     a real account plus an unknown name
    wrong_locator     a real quote cited to another place in the source
    bad_locator       a locator typed with brackets
    retyped_quote     the quote re-typed with a word dropped
    invented_quote    a quote nobody said
    twin              a second, similar fact from the same extraction call
    word_sentiment    "negative" where the schema wants an integer

Every emitted fact carries a tag in its claim, so the run can say exactly
where each one ended up: a claim, a review row, or nowhere.

    createdb portolan_test_stress
    DATABASE_URL=postgresql://localhost/portolan_test_stress STORE_ROOT=/tmp/stress/store \\
        uv run python scripts/bench_write/noisy_extractor.py --path corpus/out

Needs an empty database whose name starts with `portolan_test`. No model
call, no network. Prints one JSON object.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from portolan import db
from portolan.config import get_settings
from portolan.connectors.file_connector import FileConnector
from portolan.extract.schema import QUOTE_REQUIRED_TYPES
from portolan.fake_extractor import FakePlacements, build_fake_llm_client
from portolan.ingest.prepare import prepare_text
from portolan.llm.fake import FakeLLMClient
from portolan.metering import InMemorySink
from portolan.pipeline import run
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore

_TAG_RE = re.compile(r"\(ref zq\d+\)")
_SOURCE_ID_RE = re.compile(r"^source_id:\s*(\S+)", re.MULTILINE)
_QUOTE_TYPES = {str(t.value) for t in QUOTE_REQUIRED_TYPES}

KINDS = (
    ("no_entity", 0.08),
    ("unknown_only", 0.08),
    ("extra_unknown", 0.14),
    ("wrong_locator", 0.08),
    ("bad_locator", 0.06),
    ("retyped_quote", 0.10),
    ("invented_quote", 0.05),
    ("twin", 0.10),
    ("word_sentiment", 0.05),
)
_UNKNOWN_ORGS = [f"Unlisted Works {n}" for n in range(1, 7)]
_UNKNOWN_PEOPLE = [
    f"the {role} lead"
    for role in (
        "night shift",
        "biomed",
        "receiving dock",
        "pharmacy",
        "float pool",
        "sterile supply",
        "east wing",
        "weekend",
        "training",
        "informatics",
        "purchasing",
        "transport",
    )
]


class NoisyExtractor:
    """A `FakeLLMClient` responder: the clean fake's reply, damaged."""

    def __init__(self, clean: FakeLLMClient, seed: int) -> None:
        self._clean_responder = clean._responses
        self._seed = seed
        self._by_source: dict[str, dict] = {}
        self.emitted: dict[str, dict] = {}  # tag -> {source, kind, quote, locator}
        self.extract_calls = 0
        self._counter = 0

    def __call__(self, *, system, user, json_schema, model, surface):
        reply = self._clean_responder(
            system=system, user=user, json_schema=json_schema, model=model, surface=surface
        )
        if surface != "extract" or not isinstance(reply, dict):
            return reply
        self.extract_calls += 1
        match = _SOURCE_ID_RE.search(user)
        source_id = match.group(1) if match else "?"
        if source_id not in self._by_source:
            self._by_source[source_id] = self._damage(source_id, reply)
        return copy.deepcopy(self._by_source[source_id])

    def _tag(self, source_id: str, kind: str, fact: dict) -> None:
        self._counter += 1
        tag = f"(ref zq{self._counter})"
        fact["claim"] = f"{fact['claim'].rstrip('.')} {tag}."
        self.emitted[tag] = {
            "source": source_id,
            "kind": kind,
            "true_locator": fact.get("_true_locator", fact["locator"]),
        }
        fact.pop("_true_locator", None)

    def _damage(self, source_id: str, reply: dict) -> dict:
        rng = random.Random(f"{self._seed}:{source_id}")
        reply = copy.deepcopy(reply)
        facts = reply.get("facts") or []
        locators = [f["locator"] for f in facts]
        out: list[dict] = []
        for fact in facts:
            roll, kind, edge = rng.random(), "clean", 0.0
            for name, share in KINDS:
                edge += share
                if roll < edge:
                    kind = name
                    break
            fact["_true_locator"] = fact["locator"]
            twin = None
            if kind == "no_entity":
                fact["entities"] = []
            elif kind == "unknown_only":
                fact["entities"] = [rng.choice(_UNKNOWN_ORGS)]
            elif kind == "extra_unknown":
                fact["entities"] = [*fact["entities"], rng.choice(_UNKNOWN_PEOPLE)]
            elif kind == "wrong_locator" and fact.get("quote"):
                others = [loc for loc in locators if loc != fact["locator"]]
                fact["locator"] = rng.choice(others) if others else "00:59:59"
            elif kind == "bad_locator":
                fact["locator"] = f"[{fact['locator']}]"
            elif kind == "retyped_quote" and fact.get("quote") and len(fact["quote"].split()) > 5:
                words = fact["quote"].split()
                del words[rng.randrange(1, len(words) - 1)]
                fact["quote"] = " ".join(words)
            elif kind == "invented_quote" and fact["type"] in _QUOTE_TYPES:
                fact["quote"] = "we are going to cancel the whole agreement before the next quarter"
            elif kind == "twin":
                twin = copy.deepcopy(fact)
                twin["id"] = f"{fact['id']}t"
                twin["claim"] = _twin_claim(twin["claim"])
            elif kind == "word_sentiment":
                fact["sentiment"] = "negative"
            else:
                kind = (
                    "clean"
                    if kind in ("wrong_locator", "retyped_quote", "invented_quote")
                    else kind
                )
            self._tag(source_id, kind, fact)
            out.append(fact)
            if twin is not None:
                self._tag(source_id, "twin_copy", twin)
                out.append(twin)
        reply["facts"] = out
        return reply


_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": " "})


def _norm(text: str) -> str:
    """Case, glyph variants, ellipses and runs of whitespace ignored: what
    "the quote is in the source, word for word" means here."""
    return " ".join(text.translate(_FOLD).replace("...", " ").lower().split())


_TWIN_FILLER = (
    "and separately the same thing was reported a second time for the other building "
    "during the following week by a different unit on a different floor of the campus"
).split()


def _twin_claim(claim: str) -> str:
    """A second fact about a second event, worded like the first: alike
    enough (0.6 to 0.72) for a text-similarity dedup to fold the two, not
    alike enough for the fake compare to call them the same fact (0.75)."""
    base = claim.rstrip(".")
    words: list[str] = []
    for word in _TWIN_FILLER:
        words.append(word)
        candidate = f"{base} {' '.join(words)}."
        if SequenceMatcher(None, claim.lower(), candidate.lower()).ratio() < 0.72:
            return candidate
    return f"{base} {' '.join(words)}."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--path", type=Path, default=Path("corpus/out"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    settings = get_settings()
    url = settings.database_url.get_secret_value()
    if not url.rsplit("/", 1)[-1].startswith("portolan_test"):
        print("refusing: DATABASE_URL must name a portolan_test* database", file=sys.stderr)
        return 2
    os.environ.setdefault("PORTOLAN_EXTRACT_CONCURRENCY", "1")

    conn = db.connect()
    db.apply_migrations(conn)
    connector = FileConnector(args.path)
    clean = build_fake_llm_client(FakePlacements(args.path), tenant=settings.tenant)
    noisy = NoisyExtractor(clean, args.seed)
    llm = FakeLLMClient(noisy, model_key="nova-lite", tenant=settings.tenant)
    report = run(
        conn,
        connector=connector,
        llm=llm,
        embedder=FakeEmbedder(),
        registry=Registry.load(conn),
        store=LocalStore(settings.store_root),
        settings=settings,
        sink=InMemorySink(),
    )

    texts: dict[str, str] = {}
    spans: dict[str, set[str]] = {}
    for ref in connector.list_sources(None):
        prepared = prepare_text(connector.fetch(ref))
        texts[ref.source_id] = prepared.text
        spans[ref.source_id] = {s.locator for s in prepared.locators}

    with conn.cursor() as cur:
        cur.execute("SELECT claim_text, source_slug, locator, quote FROM claims")
        claims = cur.fetchall()
        cur.execute("SELECT status, kind, payload::text FROM review_queue")
        queue = cur.fetchall()
        cur.execute("SELECT source_slug, count(*) FROM claim_citations GROUP BY 1")
        citations = Counter({slug.split("--v")[0]: n for slug, n in cur.fetchall()})

    in_claims: dict[str, tuple] = {}
    for claim_text, source_slug, locator, quote in claims:
        for tag in _TAG_RE.findall(claim_text):
            in_claims[tag] = (source_slug, locator, quote)
    in_queue: set[str] = set()
    for _status, _kind, payload in queue:
        in_queue.update(_TAG_RE.findall(payload))

    emitted = noisy.emitted
    held_only = {t for t in in_queue if t not in in_claims}
    # A fact the compare call judged the same as a claim on file is kept as
    # a citation on that claim: one unplaced fact per citation its source wrote.
    lost: list[str] = []
    absorbed = 0
    for tag in emitted:
        if tag in in_claims or tag in in_queue:
            continue
        source = emitted[tag]["source"]
        if citations[source] > 0:
            citations[source] -= 1
            absorbed += 1
        else:
            lost.append(tag)
    lost_by_kind: dict[str, int] = {}
    for tag in lost:
        lost_by_kind[emitted[tag]["kind"]] = lost_by_kind.get(emitted[tag]["kind"], 0) + 1

    dead_locator = wrong_place = not_verbatim = quoted = invented_live = 0
    for tag, (source_slug, locator, quote) in in_claims.items():
        base = source_slug.split("--v")[0]
        if spans.get(base) and locator not in spans[base]:
            dead_locator += 1
        elif locator != emitted[tag]["true_locator"]:
            wrong_place += 1
        if quote:
            quoted += 1
            if _norm(quote) not in _norm(texts.get(base, "")):
                not_verbatim += 1
                if emitted[tag]["kind"] == "invented_quote":
                    invented_live += 1

    by_status: dict[str, int] = {}
    open_by_kind: dict[str, int] = {}
    for status, kind, _payload in queue:
        by_status[status] = by_status.get(status, 0) + 1
        if status == "open":
            open_by_kind[kind] = open_by_kind.get(kind, 0) + 1

    kinds: dict[str, int] = {}
    for info in emitted.values():
        kinds[info["kind"]] = kinds.get(info["kind"], 0) + 1

    print(
        json.dumps(
            {
                "seed": args.seed,
                "sources_extracted": report.sources_extracted,
                "sources_failed": report.sources_failed,
                "facts_emitted": len(emitted),
                "emitted_by_kind": dict(sorted(kinds.items())),
                "facts_as_claims": len(in_claims),
                "facts_held_for_review_only": len(held_only),
                "facts_kept_as_a_citation_on_an_existing_claim": absorbed,
                "facts_lost": len(lost),
                "lost_by_kind": dict(sorted(lost_by_kind.items())),
                "claims_citing_a_locator_not_in_the_source": dead_locator,
                "claims_citing_the_wrong_place": wrong_place,
                "claims_with_a_quote": quoted,
                "claim_quotes_not_verbatim_in_source": not_verbatim,
                "invented_quotes_that_became_claims": invented_live,
                "review_rows_by_status": dict(sorted(by_status.items())),
                "open_review_rows_by_kind": dict(sorted(open_by_kind.items())),
                "extraction_calls": noisy.extract_calls,
                "provenance_counts": getattr(report, "provenance_counts", None),
            },
            indent=2,
        )
    )
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
