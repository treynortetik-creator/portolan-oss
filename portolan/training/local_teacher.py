"""Local open-weight teachers: small models run on this Mac in-process with
MLX (`portolan.llm.mlx_local.MLXClient`) in place of a paid OpenRouter call
(`01-spec/local-teachers-bakeoff-2026-09-22.md`).

A local teacher id is `local:<Hugging Face repo id>`, e.g.
`local:mlx-community/Qwen3.5-4B-MLX-4bit`. That full string is what a
silver record's `teachers`, a spend line and a v2 writer call record carry,
so a label always names the exact weights that wrote it.

`LOCAL_MODELS` is the runtime table: explicit repo ids, each pinned to a Hub
revision and noted with the license that decides whether its outputs may
train a model. `require_local_teacher` refuses an id that is not listed or
whose license does not permit it. Ids match case-insensitively, and a client
always loads the table's spelling (the Hugging Face cache directory is named
after it). The training-set builder keeps its own explicit copy of the
permitted ids (`scripts/data/build_training_set.py` `ALLOWED_LOCAL_TEACHERS`);
`tests/test_local_teacher.py` pins the two together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "LOCAL_PREFIX",
    "LOCAL_MODELS",
    "LocalModel",
    "canonical_local_teacher",
    "is_local_teacher",
    "local_model_id",
    "local_teacher_id",
    "permitted_local_teacher_ids",
    "require_local_teacher",
    "make_local_client",
]

LOCAL_PREFIX = "local:"


@dataclass(frozen=True)
class LocalModel:
    """One MLX build of an open-weight model, pinned to a Hub revision, and
    the license of its base model."""

    base_model: str
    revision: str
    license: str
    license_url: str
    # Whether the license lets this model's outputs train another model.
    outputs_may_train: bool
    note: str = ""
    # Extra tokenizer settings `mlx_lm.load` passes to transformers.
    tokenizer_config: tuple[tuple[str, Any], ...] = ()


# Keys are the Hugging Face repo ids of the MLX 4-bit builds. Licenses read
# on 2026-09-22 from each base model's card and LICENSE file (and, for Gemma
# 4, Google's license page); revisions are the Hub commits benchmarked in
# 01-spec/local-teachers-bakeoff-2026-09-22.md. A new entry is a provenance
# decision (record it in 01-spec/decisions-log.md).
LOCAL_MODELS: dict[str, LocalModel] = {
    "mlx-community/Qwen3.5-4B-MLX-4bit": LocalModel(
        base_model="Qwen/Qwen3.5-4B",
        revision="32f3e8ecf65426fc3306969496342d504bfa13f3",
        license="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
        outputs_may_train=True,
        note="thinking is on by default in its template; MLXClient renders it off",
    ),
    "mlx-community/gemma-4-e4b-it-4bit": LocalModel(
        base_model="google/gemma-4-e4b-it",
        revision="475b9088d29754a3379866cf5aeb6b41acd313c2",
        license="Apache-2.0",
        license_url="https://ai.google.dev/gemma/docs/gemma_4_license",
        outputs_may_train=True,
        note=(
            "the MLX card's front matter says `license: gemma`; the base model card and "
            "Google's Gemma 4 license page both say Apache 2.0, which governs the weights"
        ),
    ),
    "mlx-community/Ministral-3-8B-Instruct-2512-4bit": LocalModel(
        base_model="mistralai/Ministral-3-8B-Instruct-2512",
        revision="182f003f01daa75f9de0f2c4d379722fd0bc1c61",
        license="Apache-2.0",
        license_url="https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512",
        outputs_may_train=True,
        note=(
            "text only: mlx_lm drops the vision tower on load; the MLX build's tokenizer "
            "carries the pre-tokenizer regex transformers warns is wrong for Mistral, so "
            "it loads with fix_mistral_regex"
        ),
        tokenizer_config=(("fix_mistral_regex", True),),
    ),
    "mlx-community/Phi-4-mini-instruct-4bit": LocalModel(
        base_model="microsoft/Phi-4-mini-instruct",
        revision="ac1c269cb4222a4e136a3d09edad301056c1f36a",
        license="MIT",
        license_url="https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/main/LICENSE",
        outputs_may_train=True,
    ),
}


def is_local_teacher(teacher: str) -> bool:
    return teacher.strip().lower().startswith(LOCAL_PREFIX)


def local_model_id(teacher: str) -> str:
    """The Hugging Face repo id inside a `local:` teacher id, as given."""
    if not is_local_teacher(teacher):
        raise ValueError(
            f"{teacher!r} is not a local teacher id (expected {LOCAL_PREFIX}<repo id>)"
        )
    repo_id = teacher.strip()[len(LOCAL_PREFIX) :].strip()
    if not repo_id or "/" not in repo_id:
        raise ValueError(f"{teacher!r}: expected {LOCAL_PREFIX}<owner>/<model>")
    return repo_id


def local_teacher_id(repo_id: str) -> str:
    return f"{LOCAL_PREFIX}{repo_id}"


def _table_entry(repo_id: str) -> tuple[str, LocalModel] | None:
    wanted = repo_id.lower()
    for key, info in LOCAL_MODELS.items():
        if key.lower() == wanted:
            return key, info
    return None


def permitted_local_teacher_ids() -> frozenset[str]:
    """`local:` ids (table spelling) whose license permits training on outputs."""
    return frozenset(
        local_teacher_id(repo_id)
        for repo_id, info in LOCAL_MODELS.items()
        if info.outputs_may_train
    )


def require_local_teacher(teacher: str) -> LocalModel:
    """The table entry for `teacher`, or `ValueError` when it is not listed
    or its license does not permit training on its outputs."""
    repo_id = local_model_id(teacher)
    entry = _table_entry(repo_id)
    if entry is None:
        raise ValueError(
            f"local teacher {repo_id!r} is not in portolan.training.local_teacher.LOCAL_MODELS; "
            "add it with its license first"
        )
    info = entry[1]
    if not info.outputs_may_train:
        raise ValueError(
            f"local teacher {repo_id!r}: its license ({info.license}) does not permit "
            "training on outputs"
        )
    return info


def canonical_local_teacher(teacher: str) -> str:
    """`teacher` spelled as in the table, after `require_local_teacher`."""
    require_local_teacher(teacher)
    entry = _table_entry(local_model_id(teacher))
    assert entry is not None
    return local_teacher_id(entry[0])


def make_local_client(teacher: str, **kwargs: Any) -> Any:
    """An `MLXClient` for a permitted `local:` teacher at its pinned
    revision. MLX is imported lazily by the client, so this never needs it
    until the first call."""
    from portolan.llm.mlx_local import MLXClient

    info = require_local_teacher(teacher)
    kwargs.setdefault("revision", info.revision)
    kwargs.setdefault("tokenizer_config", dict(info.tokenizer_config))
    return MLXClient(local_model_id(canonical_local_teacher(teacher)), **kwargs)
