"""Manifest/truth-table-driven fake extraction (build-contracts.md
"Pipeline runner" item 5: "`--llm fake` uses a `FakeLLMClient` whose
extraction responses are generated from corpus/out/manifest.json
placements... build that helper in `portolan/pipeline.py` or a small
`portolan/fake_extractor.py` you own").

This is what makes `python -m portolan.pipeline --llm fake` produce
exactly the facts `corpus/truth-table.yaml` planted, with no model call,
so the end-to-end path is testable without a live LLM.

Two files feed this:
- `corpus/truth-table.yaml` — the answer key. Its `facts:` list gives
  each planted fact's `type`, `claim`, `entities` (truth-table slugs),
  `date`, and, per fact, which sources it was planted in
  (`sources: [{id, type, speaker, position?}, ...]`) — this is the primary
  index: "for source id X, which facts (if any) belong to it."
- `corpus/out/manifest.json` — gives the exact `locator`/`excerpt` string
  for a given `(fact_id, source_id)` pair (the truth table's own
  `sources[]` entries don't carry a locator, only a speaker/position).
  Optional: when absent, a locator is synthesised as `<source_id>#
  <fact_id>` — still a valid, non-empty `Fact.locator`, just not tied to
  a real span in the source text (fine for a fake path; nothing here
  checks a quote against the real text either).

Entity resolution: `Fact.entities`/`EntityMention.registry_id` need real
registry ids, not truth-table slugs (`portolan/registry/registry.py`'s
module docstring: "Truth-table account slugs are aliases only"). This
module builds slug -> registry id maps the same way
`tests/ingest_helpers.py`'s `build_fake_registry` does: an account or
contact slug maps through its own `crm_id`; a product/competitor/
internal-person slug is already the registry id verbatim (the meta
block's own `id`, imported as-is by `Registry.import_crm`).

Segmentation: when `portolan/pipeline.py` splits a long call transcript
into multiple ~4k-token segments, each segment gets its own extraction
call and `portolan.extract.segment.merge_segment_results` recombines
them. Rather than track which planted fact's locator falls inside which
segment's character range, this responder takes the simpler, equally
correct route: it returns *every* fact planted for that `source_id` on
*every* segment call. `merge_segment_results`'s own per-source dedup
(same type, overlapping entities, matching locator/claim) collapses the
duplicates back down to one occurrence each — so the merged, per-source
result is identical to what a real single-shot call would have produced,
without this module needing to reason about character offsets at all.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from portolan.extract.schema import (
    QUOTE_REQUIRED_TYPES,
    Action,
    Coverage,
    EntityKind,
    EntityMention,
    ExtractionResult,
    Fact,
    FactType,
)
from portolan.llm.fake import FakeLLMClient
from portolan.metering import Surface, UsageSink

__all__ = ["FakePlacements", "build_fake_llm_client"]

# `portolan.extract.prompt.build_user_message` always starts the user
# message with this exact line -- see that module's docstring/source.
_SOURCE_ID_RE = re.compile(r"^source_id:\s*(\S+)", re.MULTILINE)

# Shared plain-text normaliser (review finding, this fix wave): strips
# `[[slug]]` wikilinks down to space-separated words and collapses
# everything else to lowercase alphanumerics-plus-spaces. Used both to
# match a compare prompt's rendered claim text back to the truth table's
# planted fact it came from (`FakePlacements._build_claim_lookup`,
# `_best_fact_match`) and, as before, to score two claim texts' similarity
# (`_compare_verdict_for`). Defined once, ahead of `FakePlacements`, so
# `__init__` can build its lookup table with the exact same normalisation
# the compare path applies later.
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")


def _plain_for_compare(text: str) -> str:
    text = _WIKILINK_RE.sub(lambda m: m.group(1).replace("-", " "), text)
    return _NON_ALNUM_RE.sub(" ", text.lower())


# A small, deliberately simple sentiment-by-type default -- not planted
# data (the truth table has no per-fact sentiment field), just enough to
# satisfy `Fact.sentiment`'s required `-2..2` range with something
# directionally sane.
_SENTIMENT_BY_TYPE: dict[FactType, int] = {
    FactType.COMPLAINT: -1,
    FactType.RISK: -2,
    FactType.PRAISE: 1,
    FactType.COMMITMENT: 1,
    FactType.CONTACT_CHANGE: 0,
    FactType.COMPETITOR_MENTION: -1,
    FactType.DECISION: 0,
    FactType.PRICING: 0,
    FactType.USAGE: 0,
    FactType.QUESTION: 0,
    FactType.REQUEST: 0,
    FactType.OTHER: 0,
}


class FakePlacements:
    """Precomputed lookup built once from `corpus/truth-table.yaml` (facts,
    entities) and `corpus/out/manifest.json` (locators), answering:
    "what does source_id X's `ExtractionResult` look like?"
    """

    def __init__(self, corpus_root: Path | str, truth_table_path: Path | str | None = None) -> None:
        self.corpus_root = Path(corpus_root)

        if truth_table_path is not None:
            tt_path = Path(truth_table_path)
        else:
            # corpus/out -> corpus/truth-table.yaml
            tt_path = self.corpus_root.parent / "truth-table.yaml"
        self.truth_table: dict[str, Any] = {}
        if tt_path.is_file():
            self.truth_table = yaml.safe_load(tt_path.read_text(encoding="utf-8")) or {}

        manifest_path = self.corpus_root / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self._registry_id, self._kind, self._name = self._build_entity_maps()
        self._locator_by_fact_source = self._build_locator_index(manifest)
        self._facts_by_source_id = self._build_facts_by_source()

        # Review finding, this fix wave: `fake_extractor` never emitted
        # `contradicts` verdicts or `Action`s, so `merge_action` and the
        # whole supersession path had zero end-to-end coverage from a fake
        # pipeline run. These three indexes are what fixes both:
        # `claim_lookup`/`supersedes` let `_compare_verdict_for` (module
        # level, below) recognise the truth table's two planted reversals
        # (F11 -> F11b, F15 -> F15b); `_actions_open` is what
        # `extraction_result_for` reads to emit the F09/F32/F33 `Action`s
        # `expectations.report.actions_open` names.
        self.claim_lookup = self._build_claim_lookup()
        self.supersedes = self._build_supersedes_map()
        self._actions_open = self._build_actions_open()
        # Decisions log 2026-09-14 ("withdraw content-based automatic
        # resolution"): what the fake resolution judge (`_resolution_
        # verdict_for`, module level, below) reads to decide `resolves`
        # -- every fact id named in ANY theme's own `resolution_facts`
        # list (`expectations.report.themes.*.resolution_facts`), the
        # truth table's own answer to "which planted facts are genuine
        # resolution evidence," the same `expectations.report.*` reading
        # pattern `_build_actions_open` already uses.
        self.resolution_fact_ids = self._build_resolution_fact_ids()

    # -- index construction ------------------------------------------------

    def _build_entity_maps(
        self,
    ) -> tuple[dict[str, str], dict[str, EntityKind], dict[str, str]]:
        registry_id: dict[str, str] = {}
        kind: dict[str, EntityKind] = {}
        name: dict[str, str] = {}

        for account in self.truth_table.get("accounts", []) or []:
            slug = account["id"]
            registry_id[slug] = account.get("crm_id") or slug
            kind[slug] = EntityKind.ACCOUNT
            name[slug] = account.get("name", slug)

        for contact in self.truth_table.get("contacts", []) or []:
            slug = contact["id"]
            registry_id[slug] = contact.get("crm_id") or slug
            kind[slug] = EntityKind.CONTACT
            name[slug] = contact.get("name", slug)

        meta = self.truth_table.get("meta", {}) or {}
        for product in meta.get("products", []) or []:
            slug = product["id"]
            registry_id[slug] = slug
            kind[slug] = EntityKind.PRODUCT
            name[slug] = product.get("name", slug)
        for competitor in meta.get("competitors", []) or []:
            slug = competitor["id"]
            registry_id[slug] = slug
            kind[slug] = EntityKind.COMPETITOR
            name[slug] = competitor.get("name", slug)
        for person in meta.get("internal_people", []) or []:
            slug = person["id"]
            registry_id[slug] = slug
            kind[slug] = EntityKind.INTERNAL_PERSON
            name[slug] = person.get("name", slug)

        return registry_id, kind, name

    def _build_locator_index(
        self, manifest: dict[str, Any]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        index: dict[tuple[str, str], dict[str, Any]] = {}
        for source in manifest.get("sources", []) or []:
            source_id = source.get("id")
            for fact in source.get("facts", []) or []:
                index[(fact["fact_id"], source_id)] = fact
        return index

    def _build_facts_by_source(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for fact_def in self.truth_table.get("facts", []) or []:
            # `truth-table.yaml`'s top-level `facts:` list also carries the
            # "Distractors" block (`type: distractor`, ids `D01`..`D04`) --
            # things that look like facts and, per the truth table's own
            # `expectations.extraction.distractors_must_not_yield_facts`,
            # must NOT be extracted. Skip them here so a fake run doesn't
            # invent facts on the sources planted specifically to test
            # that nothing gets invented (e.g. `S-call-002`, `S-email-040`).
            if fact_def.get("type") == "distractor":
                continue
            for source_entry in fact_def.get("sources", []) or []:
                result.setdefault(source_entry["id"], []).append(
                    {"fact_def": fact_def, "source_entry": source_entry}
                )
        return result

    def _build_claim_lookup(self) -> dict[str, str]:
        """`{plain_normalised_claim_text: fact_id}` for every planted
        (non-distractor) fact -- how `best_fact_match` identifies which
        truth-table fact a rendered claim/compare-prompt text came from."""
        lookup: dict[str, str] = {}
        for fact_def in self.truth_table.get("facts", []) or []:
            if fact_def.get("type") == "distractor":
                continue
            lookup[_plain_for_compare(fact_def["claim"])] = fact_def["id"]
        return lookup

    def _build_supersedes_map(self) -> dict[str, str]:
        """`{new_fact_id: old_fact_id}` straight off the truth table's own
        `supersedes:` field (F11b -> F11, F15b -> F15) -- the ground truth
        `_compare_verdict_for` checks to decide "contradicts"."""
        return {
            fact_def["id"]: fact_def["supersedes"]
            for fact_def in (self.truth_table.get("facts", []) or [])
            if fact_def.get("supersedes")
        }

    def _build_actions_open(self) -> dict[str, dict[str, Any]]:
        """`{fact_id: {"owner": <truth-table person slug or None>, "due":
        <date or None>}}` off `expectations.report.actions_open`
        (`corpus/truth-table.yaml`): the truth table's own answer to
        "which facts should also produce an open action" -- F09 (owner,
        due date), F32 (owner, no due date), F33 (a customer commitment,
        no internal owner)."""
        report_expectations = (self.truth_table.get("expectations") or {}).get("report") or {}
        result: dict[str, dict[str, Any]] = {}
        for entry in report_expectations.get("actions_open") or []:
            fact_id = entry.get("from")
            if fact_id:
                result[fact_id] = entry
        return result

    def _build_resolution_fact_ids(self) -> set[str]:
        """Every fact id named in any theme's own `resolution_facts` list
        (`expectations.report.themes.*.resolution_facts`) -- see this
        field's own comment on `__init__`."""
        report_expectations = (self.truth_table.get("expectations") or {}).get("report") or {}
        themes = report_expectations.get("themes") or {}
        result: set[str] = set()
        for theme in themes.values():
            result.update(theme.get("resolution_facts") or [])
        return result

    def best_fact_match(self, plain_text: str, *, threshold: float) -> str | None:
        """The truth-table fact id whose own (plain-normalised) claim text
        is the closest `difflib` match to `plain_text`, or `None` if
        nothing clears `threshold`."""
        best_id: str | None = None
        best_ratio = 0.0
        for plain_claim, fact_id in self.claim_lookup.items():
            ratio = SequenceMatcher(None, plain_text, plain_claim).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = fact_id
        return best_id if best_ratio >= threshold else None

    # -- the answer ---------------------------------------------------------

    def has_placements(self, source_id: str) -> bool:
        return bool(self._facts_by_source_id.get(source_id))

    def extraction_result_for(self, source_id: str) -> ExtractionResult:
        entries = self._facts_by_source_id.get(source_id, [])
        facts: list[Fact] = []
        actions: list[Action] = []
        entities: list[EntityMention] = []
        seen_entities: set[tuple[str, str]] = set()

        for index, entry in enumerate(entries, start=1):
            fact_def = entry["fact_def"]
            source_entry = entry["source_entry"]
            fact_id = fact_def["id"]

            try:
                fact_type = FactType(fact_def["type"])
            except ValueError:
                fact_type = FactType.OTHER

            claim = fact_def["claim"]
            locator_info = self._locator_by_fact_source.get((fact_id, source_id))
            locator = locator_info["locator"] if locator_info else f"{source_id}#{fact_id}"

            quote: str | None = None
            if fact_type in QUOTE_REQUIRED_TYPES:
                quote_source = locator_info["excerpt"] if locator_info else claim
                quote = " ".join(quote_source.split()[:40])

            valid_from = None
            raw_date = fact_def.get("date")
            if raw_date:
                try:
                    from datetime import date as _date

                    valid_from = _date.fromisoformat(str(raw_date))
                except ValueError:
                    valid_from = None

            entity_ids: list[str] = []
            for slug in fact_def.get("entities", []) or []:
                registry_id = self._registry_id.get(slug, slug)
                entity_ids.append(registry_id)
                entity_kind = self._kind.get(slug, EntityKind.OTHER)
                key = (slug, entity_kind.value)
                if key not in seen_entities:
                    seen_entities.add(key)
                    entities.append(
                        EntityMention(
                            mention=self._name.get(slug, slug),
                            kind=entity_kind,
                            registry_id=registry_id,
                            confidence=0.95,
                        )
                    )

            facts.append(
                Fact(
                    id=f"f{index}",
                    type=fact_type,
                    claim=claim,
                    entities=entity_ids,
                    locator=locator,
                    quote=quote,
                    speaker=source_entry.get("speaker"),
                    valid_from=valid_from,
                    valid_to=None,
                    severity=None,
                    sentiment=_SENTIMENT_BY_TYPE.get(fact_type, 0),
                    confidence=0.9,
                )
            )

            # F09/F32/F33 (`expectations.report.actions_open`): this fact
            # also produces an open `Action`, carrying the same locator the
            # fact itself just got -- an action is "not a fact" but still
            # needs a citation (extraction-schema-v0.md "Actions").
            action_info = self._actions_open.get(fact_id)
            if action_info is not None:
                owner_slug = action_info.get("owner")
                owner_mention = self._name.get(owner_slug, owner_slug) if owner_slug else None
                actions.append(
                    Action(
                        what=claim,
                        owner_mention=owner_mention,
                        due=action_info.get("due"),
                        locator=locator,
                    )
                )

        summary = facts[0].claim if facts else f"No planted facts for {source_id}."

        return ExtractionResult(
            source_id=source_id,
            summary=summary,
            entities=entities,
            facts=facts,
            actions=actions,
            unknown_entities=[],
            coverage=Coverage(segments_read=1, segments_total=1),
        )


def _extract_source_id(user_message: str) -> str | None:
    match = _SOURCE_ID_RE.search(user_message)
    return match.group(1) if match else None


# --------------------------------------------------------------------------
# portolan.merge's compare call (surface="merge").
#
# `portolan.merge.gate.build_compare_prompt` (wave B1) renders the compare
# call's user message in one fixed shape:
#
#     Existing claim on file, recorded YYYY-MM-DD, cited to <slug> @
#     <locator>:
#     <existing claim text>
#
#     New fact, locator <locator>[, dated YYYY-MM-DD]:
#     <new fact claim text>
#
# and asks (`portolan.merge.gate.COMPARE_SYSTEM`) for a `CompareVerdict`
# (`{"verdict": "same"|"extends"|"contradicts", "reason": str}`). Rather
# than guess blindly, this parses the two claim texts back out of that
# fixed shape and compares them:
#
# - First, both sides are matched back to the truth-table fact id they
#   most resemble (`FakePlacements.best_fact_match`, same `_plain_for_
#   compare` normalisation the similarity score below already used). If
#   the *new* side's fact is recorded as `supersedes: <the existing
#   side's fact id>` in `corpus/truth-table.yaml` -- exactly the truth
#   table's two planted reversals, F11 -> F11b and F15 -> F15b -- the
#   verdict is "contradicts". Before this fix (review finding, this fix
#   wave: "fake_extractor never verdicts contradicts") this module could
#   never produce that verdict at all, so supersession and the two-clocks
#   `valid_to` path had zero end-to-end coverage from a fake pipeline run.
# - Otherwise, near-identical text (the same underlying fact restated,
#   possibly from a different source -- exactly the truth-table's planted
#   cross-source duplicates, e.g. F02 in both `S-call-009` and
#   `S-ticket-031`) verdicts "same"; anything less similar verdicts
#   "extends".
# --------------------------------------------------------------------------

_COMPARE_PROMPT_RE = re.compile(
    r"cited to [^\n]*:\n(?P<existing>.*?)\n\nNew fact, locator[^\n]*:\n(?P<new>.*)",
    re.DOTALL,
)

# Empirically calibrated against the truth table's own fact pairs
# (`corpus/truth-table.yaml` `expectations.merge.cross_source_single_claim`
# / `supersessions`): identical restatements (F02, F19, F34, F11 vs F11
# itself) score 1.0; a genuine supersession (F11 -> F11b: same entities,
# different date and outcome) scores ~0.6. 0.75 sits cleanly between them.
_SAME_SIMILARITY_THRESHOLD = 0.75

# A fact match below this ratio is too weak to trust for the
# supersession lookup -- better to fall through to the plain similarity
# verdict than misidentify which planted fact a claim came from.
_FACT_MATCH_THRESHOLD = 0.5


def _compare_verdict_for(user_message: str, placements: FakePlacements) -> dict[str, str]:
    match = _COMPARE_PROMPT_RE.search(user_message)
    if not match:
        return {"verdict": "extends", "reason": "fake responder: could not parse compare prompt"}
    existing = _plain_for_compare(match.group("existing"))
    new = _plain_for_compare(match.group("new"))

    existing_fact_id = placements.best_fact_match(existing, threshold=_FACT_MATCH_THRESHOLD)
    new_fact_id = placements.best_fact_match(new, threshold=_FACT_MATCH_THRESHOLD)
    if (
        new_fact_id is not None
        and existing_fact_id is not None
        and placements.supersedes.get(new_fact_id) == existing_fact_id
    ):
        return {
            "verdict": "contradicts",
            "reason": f"fake responder: {new_fact_id} supersedes {existing_fact_id} "
            "(corpus/truth-table.yaml)",
        }

    ratio = SequenceMatcher(None, existing, new).ratio()
    verdict = "same" if ratio >= _SAME_SIMILARITY_THRESHOLD else "extends"
    return {"verdict": verdict, "reason": f"fake responder: text similarity {ratio:.2f}"}


# --------------------------------------------------------------------------
# The fake resolution judge (decisions log 2026-09-14, "withdraw content-
# based automatic resolution", task item 3). `portolan.report.themes.
# _build_resolution_judge_prompt` always ends its user message with the
# candidate claim's own plain text, quoted, on the last line -- the same
# shape `_COMPARE_PROMPT_RE` above parses for the merge compare prompt.
# `best_fact_match` (the same lookup `_compare_verdict_for` already uses,
# same `_FACT_MATCH_THRESHOLD`) identifies which planted fact the
# candidate came from; `resolves` is `True` only when that fact is one of
# the truth table's own named resolution facts
# (`FakePlacements.resolution_fact_ids`). `evidence` is the candidate's
# OWN full text -- trivially a substring of itself, satisfying `_check_
# theme_resolutions`' own exact-substring check without this fake
# responder needing to guess which clause a real model would have quoted.
# --------------------------------------------------------------------------

_RESOLUTION_CANDIDATE_RE = re.compile(
    r'Candidate claim, from [^\n]*:\n"(?P<candidate>.*)"\s*\Z',
    re.DOTALL,
)


def _resolution_verdict_for(user_message: str, placements: FakePlacements) -> dict[str, Any]:
    match = _RESOLUTION_CANDIDATE_RE.search(user_message)
    if not match:
        return {
            "resolves": False,
            "evidence": "",
            "reason": "fake responder: could not parse resolution judge prompt",
        }
    # Nit 14 (review-theme-wave-5-judge.md finding 14): `themes._build_
    # resolution_judge_prompt` now escapes `"` as `\"` before interpolating
    # customer-authored text (`themes._escape_prompt_quotes`), the same way
    # a real model reading this prompt would understand `\"` as a literal
    # quote character within the text, not the end of the quoted block --
    # unescaped back here so a planted fact's own `"..."` (F62-F65 in
    # corpus/truth-table.yaml all quote the customer verbatim) still
    # matches, and so `evidence` below is still a genuine substring of the
    # UNESCAPED candidate text `_judge_resolution`'s own check compares it
    # against.
    #
    # Nit 6 (review-theme-wave-5b.md finding 6): `_escape_prompt_quotes`
    # now also escapes a literal newline as `\n`, applied AFTER the quote
    # escape -- inverted here in the opposite order (newline first, then
    # quote) so the two substitutions round-trip regardless of which
    # character a planted fact's own text happens to contain.
    candidate_text = match.group("candidate").replace("\\n", "\n").replace('\\"', '"')
    fact_id = placements.best_fact_match(
        _plain_for_compare(candidate_text), threshold=_FACT_MATCH_THRESHOLD
    )
    if fact_id is not None and fact_id in placements.resolution_fact_ids:
        return {
            "resolves": True,
            "evidence": candidate_text,
            "reason": f"fake responder: {fact_id} is a planted resolution fact "
            "(corpus/truth-table.yaml expectations.report.themes.*.resolution_facts)",
        }
    return {
        "resolves": False,
        "evidence": "",
        "reason": "fake responder: not a planted resolution fact",
    }


# --------------------------------------------------------------------------
# Generic minimal-JSON-schema filler -- the last-resort fallback for any
# other LLM surface this module doesn't specifically know the shape of
# (chat, report prose, consolidation, ...; none of these are exercised by
# the pipeline's own tests). Fills every required property with the
# smallest value its schema allows, biasing any enum that looks like a
# merge outcome toward "same" in case a future surface reuses that shape.
# --------------------------------------------------------------------------


def _resolve_schema(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in schema:
        node: Any = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return node
    return schema


def _minimal_value_for(schema: dict[str, Any], root: dict[str, Any]) -> Any:
    schema = _resolve_schema(schema, root)

    if "enum" in schema:
        values = schema["enum"]
        for preferred in ("same", "extends"):
            if preferred in values:
                return preferred
        return values[0] if values else None

    if "default" in schema:
        return schema["default"]

    if "anyOf" in schema:
        for option in schema["anyOf"]:
            resolved = _resolve_schema(option, root)
            if resolved.get("type") != "null":
                return _minimal_value_for(option, root)
        return None

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        return _minimal_object_for(schema, root)
    if schema_type == "array":
        return []
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema_type == "boolean":
        return False
    return ""


def _minimal_object_for(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    required = schema.get("required", list(properties.keys()))
    return {
        name: _minimal_value_for(properties[name], root) for name in required if name in properties
    }


def minimal_json_for_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return _minimal_object_for(schema, schema)


def build_fake_llm_client(
    placements: FakePlacements,
    *,
    model_key: str = "nova-lite",
    sink: UsageSink | None = None,
    tenant: str = "dev",
) -> FakeLLMClient:
    """A `FakeLLMClient` whose extraction responses (`surface="extract"`)
    are exactly the facts `placements` says the source carries, parsed off
    the `source_id: ...` line every `portolan.extract.prompt.
    build_user_message` call emits. Any other surface (chat, merge's
    compare call, ...) falls back to `minimal_json_for_schema` when a
    schema was given, or a short fixed string otherwise -- see the module
    docstring for why that fallback is speculative.
    """

    def _responder(
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None,
        model: str | None,
        surface: Surface,
    ) -> Any:
        del model
        if surface == "extract" and json_schema is not None:
            source_id = _extract_source_id(user)
            if source_id is not None:
                return placements.extraction_result_for(source_id).model_dump(mode="json")
        if surface == "merge" and json_schema is not None:
            return _compare_verdict_for(user, placements)
        # Decisions log 2026-09-14 ("withdraw content-based automatic
        # resolution"): the resolution judge is also `surface="report"`,
        # the same surface theme naming/headline/standing-question prose
        # already use. Nit 12 (review-theme-wave-5-judge.md finding 12):
        # matched by the SCHEMA's own `resolves` property, not by
        # importing `portolan.report.themes.RESOLUTION_JUDGE_SYSTEM` and
        # comparing the exact system prompt string -- this module is
        # packaged, not test-only (`fake_extractor.py`'s own module
        # docstring), so binding it to the report module's internal
        # prompt constant was a needless coupling; no other `surface=
        # "report"` schema in this codebase (theme naming's `name`/
        # `synthesis`, the headline's/standing-question's `sentences`,
        # account synthesis's `synthesis`) has a `resolves` property.
        if (
            surface == "report"
            and json_schema is not None
            and "resolves" in json_schema.get("properties", {})
        ):
            return _resolution_verdict_for(user, placements)
        if json_schema is not None:
            return minimal_json_for_schema(json_schema)
        return "ok"

    return FakeLLMClient(_responder, model_key=model_key, sink=sink, tenant=tenant)
