"""Which worlds a v2 run builds (`01-spec/world-generator-v2-2026-09-22.md`
sections 6 and 8): one `WorldSpec` per world, the round-1 list, the pilot,
and the parser for `scripts/data/synth_v2.py --worlds`.

Round 1 is 12 train, 2 validation and 3 test worlds. Section 4: two train
worlds per vertical, one each for two verticals; two test worlds in known
verticals and one in the held-out vertical. Which two verticals get one
train world, and every seed, is this module's choice; the orchestrator may
edit the list before the round (the cost model and the runner both read it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from corpus.generator.v2.models import Split

__all__ = [
    "PACKS_DIR",
    "PILOT",
    "ROUND_1",
    "Strictness",
    "WorldSpec",
    "known_packs",
    "parse_worlds",
    "world_id",
]

PACKS_DIR = Path(__file__).resolve().parent / "packs"

# "test" strictness: a second teacher on every document and three-judge
# disputes (section 7). The pilot runs at test strictness (section 8).
Strictness = Literal["train", "test"]


def world_id(pack: str, seed: int) -> str:
    """`simulator.simulate_world`'s `World.id` for this pack and seed."""
    return f"world-{pack}-{seed}"


@dataclass(frozen=True)
class WorldSpec:
    pack: str
    split: Split
    seed: int
    months: int = 8
    strictness: Strictness = "train"

    @property
    def world_id(self) -> str:
        return world_id(self.pack, self.seed)

    def label(self) -> str:
        return f"{self.pack}:{self.split.value}:{self.seed}"


def _w(pack: str, split: str, seed: int, strictness: Strictness | None = None) -> WorldSpec:
    s = Split(split)
    return WorldSpec(
        pack, s, seed, strictness=strictness or ("test" if s == Split.TEST else "train")
    )


ROUND_1: tuple[WorldSpec, ...] = (
    _w("pharmacy_services", "train", 1),
    _w("pharmacy_services", "train", 2),
    _w("medical_devices", "train", 3),
    _w("medical_devices", "train", 4),
    _w("ehr_addons", "train", 5),
    _w("ehr_addons", "train", 6),
    _w("healthcare_staffing", "train", 7),
    _w("healthcare_staffing", "train", 8),
    _w("medical_supplies", "train", 9),
    _w("medical_supplies", "train", 10),
    _w("senior_living_tech", "train", 11),
    _w("telehealth", "train", 12),
    _w("telehealth", "validation", 101),
    _w("senior_living_tech", "validation", 102),
    _w("medical_devices", "test", 201),
    _w("pharmacy_services", "test", 202),
    _w("lab_and_imaging_services", "test", 203),
)

# The pilot is round 1's first world run at test strictness, so its writer
# output is reused (resumed) when the round runs.
PILOT = WorldSpec("pharmacy_services", Split.TRAIN, 1, strictness="test")


def known_packs() -> list[str]:
    return sorted(p.stem for p in PACKS_DIR.glob("*.yaml"))


def parse_worlds(
    value: str | None,
    *,
    pack: str,
    split: str,
    seed: int,
    months: int = 8,
    strictness: Strictness | None = None,
) -> list[WorldSpec]:
    """`--worlds` as the runner reads it:

    - empty: the one world named by `pack`, `split`, `seed`;
    - an integer N: N worlds of that pack and split, seeds `seed` to `seed + N - 1`;
    - `round1` or `pilot`: the lists above;
    - `pack:split:seed[,pack:split:seed...]`: exactly those worlds.

    `strictness` overrides every world's (default: test for the test split).
    Raises `ValueError` on an unknown pack or split, or a repeated world id."""
    packs = set(known_packs())

    def make(p: str, s: str, sd: int) -> WorldSpec:
        if p not in packs:
            raise ValueError(f"unknown pack {p!r}; known: {sorted(packs)}")
        base = _w(p, s, sd)
        return WorldSpec(p, base.split, sd, months, strictness or base.strictness)

    text = (value or "").strip()
    if not text:
        specs = [make(pack, split, seed)]
    elif text.isdigit():
        specs = [make(pack, split, seed + i) for i in range(int(text))]
    elif text in ("round1", "pilot"):
        chosen = ROUND_1 if text == "round1" else (PILOT,)
        specs = [
            WorldSpec(w.pack, w.split, w.seed, months, strictness or w.strictness) for w in chosen
        ]
    else:
        specs = []
        for item in text.split(","):
            parts = item.strip().split(":")
            if len(parts) != 3 or not parts[2].strip().isdigit():
                raise ValueError(f"--worlds item {item!r}: expected pack:split:seed")
            specs.append(make(parts[0].strip(), parts[1].strip(), int(parts[2])))
    ids = [w.world_id for w in specs]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"world ids repeat (a world id is pack plus seed): {repeated}")
    return specs
