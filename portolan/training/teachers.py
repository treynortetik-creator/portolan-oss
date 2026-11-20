"""Open-weight free-teacher discovery and pairing for silver labeling
(silver-labels build instructions item 2: "Discover `:free` models the
way `corpus/generator/prose.py::discover_free_models` does, but restrict
to open-weight families ... Pick two different families per document").

Discovery itself is not reimplemented here -- `discover_free_models`
(`corpus/generator/prose.py`) already does the live `GET /api/v1/models`
call, filtered to ids ending in `:free`, with no hardcoded id list. This
module only narrows that result to the eight named open-weight families
and picks pairs from it; it never talks to the network on its own.
`portolan.training.pipeline` is the caller that actually schedules calls
against a chosen pair, using `corpus.generator.prose.TokenBucketLimiter`
for the shared per-minute/per-day budget (build instructions item 4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OPEN_WEIGHT_FAMILIES",
    "family_of",
    "is_open_weight",
    "filter_open_weight",
    "discover_open_weight_free_models",
    "TeacherPairPicker",
]

# Substrings matched against an OpenRouter model id (e.g.
# "qwen/qwen3.8-27b:free"), one per allowed open-weight family (build
# instructions item 2's literal list: "Qwen, Llama, DeepSeek, Mistral,
# Kimi/Moonshot, GLM, Gemma, gpt-oss"). Order doubles as the tie-break
# `family_of` uses when an id could match more than one entry (none do
# today, since OpenRouter's own id prefixes are already organization-
# scoped -- kept a plain first-match scan rather than a length-sorted one
# for that reason).
#
# Deliberately NOT `corpus.generator.prose.PREFERRED_FREE_TEACHER_
# FAMILIES` (also six author-preference families for a different
# purpose): that list includes `nvidia/nemotron-3-*`, which is not one of
# the eight families this build instruction names -- NVIDIA's own
# Nemotron license terms are not a plain open-weight release the way
# Llama/Qwen/DeepSeek/Mistral/GLM/Gemma/gpt-oss are, so an id matching
# only that family is excluded here even though `discover_free_models`
# would happily return it.
OPEN_WEIGHT_FAMILIES: tuple[str, ...] = (
    "qwen/",
    "meta-llama/",
    "deepseek/",
    "mistralai/",
    "moonshotai/",  # Kimi / Moonshot
    "z-ai/glm",  # GLM
    "google/gemma",
    "openai/gpt-oss",
)


def family_of(model_id: str) -> str | None:
    """Which `OPEN_WEIGHT_FAMILIES` entry `model_id` belongs to, or
    `None` when it matches none of them (closed-weight or unknown --
    excluded, per the build instructions, not guessed at)."""
    for family in OPEN_WEIGHT_FAMILIES:
        if family in model_id:
            return family
    return None


def is_open_weight(model_id: str) -> bool:
    return family_of(model_id) is not None


def filter_open_weight(model_ids: Sequence[str]) -> list[str]:
    """`model_ids`, narrowed to ids from an allowed open-weight family,
    order preserved."""
    return [m for m in model_ids if is_open_weight(m)]


def discover_open_weight_free_models(
    api_key: str | None = None,
    *,
    transport: Any | None = None,
    timeout: float = 30.0,
) -> list[str]:
    """`discover_free_models` (`corpus/generator/prose.py`), narrowed to
    open-weight families. Same never-raises contract: a network failure
    or an unexpected response shape yields `[]`, not an exception."""
    from corpus.generator.prose import discover_free_models

    free_ids = discover_free_models(api_key, transport=transport, timeout=timeout)
    return filter_open_weight(free_ids)


@dataclass
class TeacherPairPicker:
    """Picks two open-weight models from **different families** for one
    document, rotating through `models` so a run spreads across several
    free models rather than always the same pair (build instructions item
    2: "Pick two different families per document").

    `next_pair()` advances a single cursor by one call: the "first" slot
    therefore round-robins through the whole `models` list over
    successive documents; "second" is the next model after "first" (
    wrapping) whose family differs, found by a forward scan that never
    touches the cursor itself -- deterministic and side-effect-free
    beyond the one-step advance, so two runs given the same `models`
    order produce the same sequence of pairs (useful for the resumability
    test: a restart with the same discovered model list resumes the same
    rotation phase, not a randomized one).
    """

    models: list[str]
    _cursor: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        families = {family_of(m) for m in self.models}
        families.discard(None)
        if len(self.models) < 2 or len(families) < 2:
            raise ValueError(
                "TeacherPairPicker needs at least two open-weight models from at "
                f"least two different families; got {self.models!r}"
            )

    def next_pair(self) -> tuple[str, str]:
        n = len(self.models)
        first_idx = self._cursor % n
        first = self.models[first_idx]
        self._cursor += 1
        first_family = family_of(first)

        for offset in range(1, n):
            candidate = self.models[(first_idx + offset) % n]
            if family_of(candidate) != first_family:
                return first, candidate

        # Every model shares one family (shouldn't happen given
        # `__post_init__`'s check, but the loop above depends on
        # `first`'s own position never being revisited, not on the
        # invariant holding forever if `models` is mutated after
        # construction) -- fall back to the next model regardless of
        # family rather than raising mid-run.
        return first, self.models[(first_idx + 1) % n]
