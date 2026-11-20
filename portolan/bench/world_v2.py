"""Read a v2 world directory (`01-spec/world-generator-v2-2026-09-22.md`
section 1, "Output per world") for the extraction bench.

What the bench needs from a world, and where it reads it:

- **Labels**, `labels.jsonl` (required). One `LabelRecord` per line
  (`corpus.generator.v2.planner.LabelRecord`): `fact_id`, `fact_type`,
  `doc_id` (the source id the span indexes), `char_start`/`char_end` into
  that source's `prepare_text` output, and `text`. Every other field is
  optional and extra fields are kept, so WP4's writer labels (`plan_id`,
  `reference_mode`, `writer_model`), WP5's silver labels (`label_source:
  silver_completeness`, `claim`, `restates_fact_id`) and WP6's noised labels
  (`noise`, `asr_in_span`, `copy_of`, `supersedes`) all load.
- **Decoys**, `decoys.jsonl` (optional; WP4 `DecoySpan` shape).
- **Source texts.** The exact text the labels index, per source:
  1. `sources.jsonl`, one object per source: `source_id`, `source_type`,
     `date`, `text`; optional `kind`, `plan_id`, `account_id` (the account
     a connector would file it under, so the wrong one for a misfiled
     document),
     `participants` (names), `noise` (op names or `{type, params}`),
     `locators` (`[{locator, start, end}]`). This is the format WP7's
     exporter should write.
  2. Otherwise `docs/*.json` in the WP4 `WrittenDoc` / WP6 `NoisedDoc`
     shape (`prepared_sources[].{source_id, source_type, text}`), dated
     from `manifest.json` (`sources[].id` or `plan_id` to `date`) or
     `world_plan.json` (`doc_plans[].id` to `date`). A source with no date
     is skipped with a reason, never guessed.
- **Fact metadata** for the directive and reference-mode breakdowns, from
  the label itself, else `world.yaml` (`facts[].directive`,
  `facts[].reference_mode`), else `truth-table.yaml` (`facts[].phrasing`).
- **Vocabulary and accounts** for the extraction envelope, from
  `world.yaml` (`vendor.products`, `vendor.competitors`, `accounts`) or
  `truth-table.yaml` (`meta.products`, `meta.competitors`, `accounts`).

Every label and decoy is checked against its source: `text ==
prepared[char_start:char_end]`. One that fails, or whose source is missing
or skipped, is excluded and counted by reason (`LoadedWorld.
excluded_labels`), so a stale world directory shows up as numbers in the
report instead of as silently wrong scores.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from portolan.extract.schema import SourceType

__all__ = [
    "LABEL_SOURCE_ORDER",
    "LoadedWorld",
    "WorldAccount",
    "WorldDecoy",
    "WorldLabel",
    "WorldSource",
    "load_world",
    "locator_region",
    "noise_type",
]

# Which label source names a labeled fact when several labels share one
# (doc_id, fact_id) (a WP5 restatement reuses the scripted fact's id).
LABEL_SOURCE_ORDER: tuple[str, ...] = ("scripted_code", "scripted_writer", "silver_completeness")

# A document kind (WP3 `DocKind`, plus names a writer may put in
# `source_type`) mapped onto the extractor's closed `SourceType` list.
_KIND_TO_SOURCE_TYPE: dict[str, SourceType] = {
    "slack": SourceType.CHAT,
    "teams": SourceType.CHAT,
    "crm_note": SourceType.CRM,
    "crm_account": SourceType.CRM,
    "meeting_notes": SourceType.DOC,
    "near_duplicate": SourceType.DOC,
    "hard_negative": SourceType.DOC,
    "off_topic": SourceType.DOC,
}


def noise_type(op: Any) -> str | None:
    """A noise operation's type: `"asr"` for `"asr"`, `{"type": "asr", ...}`
    or `"partial_or_garbled:mojibake"` alike, so document-level ops and
    WP6's per-label strings share one breakdown key."""
    if isinstance(op, dict):
        op = op.get("type")
    if not isinstance(op, str) or not op.strip():
        return None
    return op.split(":", 1)[0].strip()


def _noise_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    out: list[str] = []
    for op in value:
        t = noise_type(op)
        if t and t not in out:
            out.append(t)
    return out


class WorldLabel(BaseModel):
    """One `labels.jsonl` row. Extra fields are kept, not rejected."""

    model_config = ConfigDict(extra="allow")

    fact_id: str
    fact_type: str
    doc_id: str
    char_start: int
    char_end: int
    text: str
    source_type: str | None = None
    locator: str | None = None
    label_source: str = "scripted_code"
    account_id: str | None = None
    date: dt.date | None = None
    claim: str | None = None
    directive: str | None = None
    reference_mode: str | None = None
    noise: list[str] = Field(default_factory=list)

    @field_validator("noise", mode="before")
    @classmethod
    def _noise(cls, v: Any) -> list[str]:
        return _noise_list(v)

    @property
    def key(self) -> str:
        return f"{self.doc_id}/{self.fact_id}"

    @property
    def span(self) -> tuple[int, int]:
        return self.char_start, self.char_end


class WorldDecoy(BaseModel):
    """One `decoys.jsonl` row: a planted near-fact (spec section 4)."""

    model_config = ConfigDict(extra="allow")

    decoy_id: str
    doc_id: str
    char_start: int
    char_end: int
    text: str
    decoy_class_id: str | None = None
    near_type: str | None = None
    noise: list[str] = Field(default_factory=list)

    @field_validator("noise", mode="before")
    @classmethod
    def _noise(cls, v: Any) -> list[str]:
        return _noise_list(v)

    @property
    def span(self) -> tuple[int, int]:
        return self.char_start, self.char_end


@dataclass
class WorldSource:
    """One extractable source: the prepared text the labels index."""

    source_id: str
    source_type: SourceType
    date: dt.date
    text: str
    kind: str
    plan_id: str | None = None
    account_id: str | None = None
    participants: list[str] = field(default_factory=list)
    noise: list[str] = field(default_factory=list)
    locators: list[tuple[str, int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class WorldAccount:
    id: str
    name: str
    aliases: tuple[str, ...] = ()


@dataclass
class LoadedWorld:
    root: Path
    world_id: str
    input_format: str
    sources: dict[str, WorldSource]
    labels: list[WorldLabel]
    decoys: list[WorldDecoy]
    excluded_labels: dict[str, int] = field(default_factory=dict)
    excluded_decoys: dict[str, int] = field(default_factory=dict)
    skipped_sources: dict[str, str] = field(default_factory=dict)
    fact_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    accounts: dict[str, WorldAccount] = field(default_factory=dict)
    products: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    supplier_name: str | None = None

    def labels_for(self, source_id: str) -> list[WorldLabel]:
        return [lab for lab in self.labels if lab.doc_id == source_id]

    def decoys_for(self, source_id: str) -> list[WorldDecoy]:
        return [d for d in self.decoys if d.doc_id == source_id]

    def directive_for(self, label: WorldLabel) -> str:
        if label.directive:
            return label.directive
        return self.fact_meta.get(label.fact_id, {}).get("directive") or "none"

    def reference_mode_for(self, label: WorldLabel) -> str:
        if label.reference_mode:
            return label.reference_mode
        return self.fact_meta.get(label.fact_id, {}).get("reference_mode") or "none"

    def claim_for(self, label: WorldLabel) -> str | None:
        return label.claim or self.fact_meta.get(label.fact_id, {}).get("claim")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: not JSON ({exc})") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _to_date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return dt.date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _source_type(value: Any, kind: str | None) -> SourceType | None:
    for candidate in (value, kind):
        if not isinstance(candidate, str):
            continue
        if candidate in _KIND_TO_SOURCE_TYPE:
            return _KIND_TO_SOURCE_TYPE[candidate]
        try:
            return SourceType(candidate)
        except ValueError:
            continue
    return None


def _locators(value: Any) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for row in value or []:
        if isinstance(row, dict) and {"locator", "start", "end"} <= row.keys():
            out.append((str(row["locator"]), int(row["start"]), int(row["end"])))
    return out


def _sources_from_jsonl(path: Path, skipped: dict[str, str]) -> dict[str, WorldSource]:
    sources: dict[str, WorldSource] = {}
    for row in _read_jsonl(path):
        sid = row.get("source_id") or row.get("id")
        if not isinstance(sid, str) or not sid:
            continue
        kind = row.get("kind") if isinstance(row.get("kind"), str) else None
        stype = _source_type(row.get("source_type"), kind)
        date = _to_date(row.get("date") or row.get("source_date"))
        text = row.get("text")
        if stype is None:
            skipped[sid] = f"unknown source_type {row.get('source_type')!r}"
            continue
        if date is None:
            skipped[sid] = "no date"
            continue
        if not isinstance(text, str):
            skipped[sid] = "no text"
            continue
        sources[sid] = WorldSource(
            source_id=sid,
            source_type=stype,
            date=date,
            text=text,
            kind=kind or stype.value,
            plan_id=row.get("plan_id"),
            account_id=row.get("account_id"),
            participants=[p for p in row.get("participants") or [] if isinstance(p, str)],
            noise=_noise_list(row.get("noise")),
            locators=_locators(row.get("locators")),
        )
    return sources


def _dates_from_manifests(root: Path) -> dict[str, dt.date]:
    """`id -> date` from `manifest.json` (sources by id and by plan id) and
    `world_plan.json` (doc plans by id)."""
    dates: dict[str, dt.date] = {}
    manifest = root / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for src in data.get("sources", []) if isinstance(data, dict) else []:
            d = _to_date(src.get("date"))
            if d is None:
                continue
            for key in ("id", "source_id", "plan_id"):
                if isinstance(src.get(key), str):
                    dates.setdefault(src[key], d)
    plan = root / "world_plan.json"
    if plan.exists():
        data = json.loads(plan.read_text(encoding="utf-8"))
        for doc in data.get("doc_plans", []) if isinstance(data, dict) else []:
            d = _to_date(doc.get("date"))
            if d is not None and isinstance(doc.get("id"), str):
                dates.setdefault(doc["id"], d)
    return dates


def _sources_from_docs(root: Path, skipped: dict[str, str]) -> dict[str, WorldSource]:
    dates = _dates_from_manifests(root)
    sources: dict[str, WorldSource] = {}
    for path in sorted((root / "docs").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            continue
        plan_id = doc.get("plan_id") or path.stem
        if doc.get("status", "written") != "written":
            skipped[plan_id] = f"doc status {doc.get('status')!r}"
            continue
        kind = doc.get("kind") if isinstance(doc.get("kind"), str) else None
        doc_noise = _noise_list(doc.get("noise"))
        # The account a connector would see: for a misfiled document (WP6),
        # the wrong one, which is what the misfiled noise measures.
        account = doc.get("filed_account_id") or doc.get("account_id") or doc.get("true_account_id")
        for rec in doc.get("prepared_sources") or []:
            sid = rec.get("source_id")
            if not isinstance(sid, str):
                continue
            stype = _source_type(rec.get("source_type"), kind)
            date = dates.get(sid) or dates.get(plan_id)
            if stype is None:
                skipped[sid] = f"unknown source_type {rec.get('source_type')!r}"
                continue
            if date is None:
                skipped[sid] = "no date (docs/*.json carry none; add manifest.json)"
                continue
            sources[sid] = WorldSource(
                source_id=sid,
                source_type=stype,
                date=date,
                text=str(rec.get("text") or ""),
                kind=kind or stype.value,
                plan_id=plan_id,
                account_id=account,
                noise=list(doc_noise),
            )
    return sources


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _names(rows: Any) -> list[str]:
    out: list[str] = []
    for row in rows or []:
        if isinstance(row, dict):
            for name in [row.get("name"), *(row.get("aliases") or [])]:
                if isinstance(name, str) and name and name not in out:
                    out.append(name)
    return out


def _check_span(text: str, start: int, end: int, expected: str) -> str | None:
    if not (0 <= start < end <= len(text)):
        return "span_out_of_range"
    if text[start:end] != expected:
        return "span_text_mismatch"
    return None


def load_world(root: Path | str) -> LoadedWorld:
    root = Path(root)
    labels_path = root / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(f"{root} has no labels.jsonl")

    skipped: dict[str, str] = {}
    if (root / "sources.jsonl").exists():
        input_format = "sources.jsonl"
        sources = _sources_from_jsonl(root / "sources.jsonl", skipped)
    elif (root / "docs").is_dir():
        input_format = "docs"
        sources = _sources_from_docs(root, skipped)
    else:
        raise FileNotFoundError(f"{root} has neither sources.jsonl nor docs/*.json")

    world_yaml = _load_yaml(root / "world.yaml")
    truth_yaml = _load_yaml(root / "truth-table.yaml")

    fact_meta: dict[str, dict[str, Any]] = {}
    for f in truth_yaml.get("facts") or []:
        if isinstance(f, dict) and isinstance(f.get("id"), str):
            fact_meta[f["id"]] = {"directive": f.get("phrasing"), "claim": f.get("claim")}
    for f in world_yaml.get("facts") or []:
        if isinstance(f, dict) and isinstance(f.get("id"), str):
            fact_meta[f["id"]] = {
                "directive": f.get("directive"),
                "reference_mode": f.get("reference_mode"),
                "claim": f.get("claim"),
            }

    accounts: dict[str, WorldAccount] = {}
    for a in world_yaml.get("accounts") or truth_yaml.get("accounts") or []:
        if isinstance(a, dict) and isinstance(a.get("id"), str):
            accounts[a["id"]] = WorldAccount(
                a["id"], str(a.get("name") or a["id"]), tuple(a.get("aliases") or ())
            )

    vendor = world_yaml.get("vendor") or {}
    meta = truth_yaml.get("meta") or {}
    products = _names(vendor.get("products") or meta.get("products"))
    competitors = _names(vendor.get("competitors") or meta.get("competitors"))
    supplier_name = vendor.get("name") or (meta.get("supplier") or {}).get("name")

    excluded_labels: dict[str, int] = {}
    labels: list[WorldLabel] = []
    for row in _read_jsonl(labels_path):
        try:
            label = WorldLabel.model_validate(row)
        except ValueError:
            excluded_labels["malformed"] = excluded_labels.get("malformed", 0) + 1
            continue
        source = sources.get(label.doc_id)
        reason = (
            ("source_skipped" if label.doc_id in skipped else "source_missing")
            if source is None
            else _check_span(source.text, label.char_start, label.char_end, label.text)
        )
        if reason is not None:
            excluded_labels[reason] = excluded_labels.get(reason, 0) + 1
            continue
        labels.append(label)

    excluded_decoys: dict[str, int] = {}
    decoys: list[WorldDecoy] = []
    if (root / "decoys.jsonl").exists():
        for row in _read_jsonl(root / "decoys.jsonl"):
            try:
                decoy = WorldDecoy.model_validate(row)
            except ValueError:
                excluded_decoys["malformed"] = excluded_decoys.get("malformed", 0) + 1
                continue
            source = sources.get(decoy.doc_id)
            reason = (
                "source_missing"
                if source is None
                else _check_span(source.text, decoy.char_start, decoy.char_end, decoy.text)
            )
            if reason is not None:
                excluded_decoys[reason] = excluded_decoys.get(reason, 0) + 1
                continue
            decoys.append(decoy)

    manifest_world = None
    if (root / "manifest.json").exists():
        data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest_world = data.get("world_id") if isinstance(data, dict) else None
    world_id = str(world_yaml.get("id") or manifest_world or root.name)

    return LoadedWorld(
        root=root,
        world_id=world_id,
        input_format=input_format,
        sources=sources,
        labels=labels,
        decoys=decoys,
        excluded_labels=excluded_labels,
        excluded_decoys=excluded_decoys,
        skipped_sources=skipped,
        fact_meta=fact_meta,
        accounts=accounts,
        products=products,
        competitors=competitors,
        supplier_name=supplier_name if isinstance(supplier_name, str) else None,
    )


_LOCATOR_TOKEN_RE = re.compile(r"^\[?@?\s*(.*?)\s*\]?$")


def locator_region(source: WorldSource, locator: str | None) -> tuple[int, int] | None:
    """The character region of `source.text` a produced locator names: from
    the source's own locator index when it has one, else the `[<locator>]`
    marker `prepare_text` injects, up to the next line that opens a marker."""
    if not locator:
        return None
    bare = _LOCATOR_TOKEN_RE.match(locator.strip())
    loc = bare.group(1) if bare else locator.strip()
    if not loc:
        return None
    for name, start, end in source.locators:
        if name == loc:
            return start, end
    marker = f"[{loc}]"
    idx = source.text.find(marker)
    if idx == -1:
        return None
    nxt = source.text.find("\n[", idx + len(marker))
    return idx, (nxt if nxt != -1 else len(source.text))
