"""`MLXClient`: an `LLMClient` that runs a local student model in-process
with Apple's `mlx-lm` (`01-spec/finetune-shakedown-2026-09-22.md`).

It exists so the extraction bench can grade a base model and the same
model plus a LoRA adapter trained by `scripts/train/mlx_lora.py` exactly
like any hosted backend: `python -m portolan.bench extract --backend mlx
--model mlx-community/Qwen3-1.7B-4bit [--adapter-path var/models/<run>/adapters]`.
The bench wraps this client in `portolan.extract.backends.LLMClientBackend`,
so the output goes through the same strict-attempt, one-retry, then
`ExtractionResult.parse_lenient` chain every other backend uses. The same
client is the local teacher behind `--teacher local:<repo id>` in silver
labeling and the v2 writer (`portolan.training.local_teacher`).

**Prompt.** A local model has no forced tool call, so the schema reaches
it the way it reaches every other non-tool client: the extractor's system
prompt plus `JSON_INSTRUCTION_TEMPLATE` with the ExtractionResult JSON
schema (`effective_system_prompt`, the one function both this client and
the training script call, so training and inference see the same bytes).
The chat template is the model's own, with thinking disabled.

**Shared-prefix cache.** The system prompt plus schema is long and
identical for every call, so this client encodes it once and
reuses its key/value cache for every source (trimmed back after each
call; a hybrid or sliding-window model's cache cannot be trimmed, so each
call runs on a copy). A template with no stand-alone system turn caches
the prefix its renders share instead. For a standard adapter this is a
pure speed-up with the same output. For an adapter trained in `frozen_base` prefix mode (the M1 mode
of `scripts/train/mlx_lora.py`, read from the adapter's
`adapter_config.json`), the prefix is encoded with the adapter switched
off, which is exactly how that adapter was trained.

**Dependencies.** `mlx`/`mlx-lm` live in the optional `mlx` dependency
group and are imported lazily, inside methods, so importing this module
never needs them (CI and the Docker image do not install them). The pure
helpers at the top (`effective_system_prompt`, `JsonObjectStop`,
`read_prefix_mode`) are testable without MLX.
"""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import re
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from portolan.llm.base import LLMResult, resolve_temperature
from portolan.llm.json_util import extract_json_object, validate_json_object
from portolan.llm.openrouter import JSON_INSTRUCTION_TEMPLATE
from portolan.metering import InMemorySink, Surface, UsageSink, metered

__all__ = [
    "PREFIX_MODE_FROZEN_BASE",
    "PREFIX_MODE_STANDARD",
    "RETRY_MARKER",
    "JsonObjectStop",
    "MLXClient",
    "effective_system_prompt",
    "lora_disabled",
    "mlx_available",
    "read_prefix_mode",
    "strip_leading_think",
]

#: Adapter trained on the full sequence (ordinary LoRA): apply it everywhere.
PREFIX_MODE_STANDARD = "standard"
#: Adapter trained with the shared system prefix encoded by the frozen base
#: model (adapter off) -- `scripts/train/mlx_lora.py`'s M1 mode.
PREFIX_MODE_FROZEN_BASE = "frozen_base"

#: The opening words of `portolan.extract.backends._build_retry_message`'s
#: appended block. Only used to label a metering event as a retry for the
#: shakedown's call counts; nothing branches on it.
RETRY_MARKER = "Your previous response did not validate against the required JSON"

_PREFILL_CHUNK = 2048


def mlx_available() -> bool:
    """True when `mlx` and `mlx_lm` are importable (Apple Silicon with
    `uv sync --group mlx`)."""
    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def effective_system_prompt(system: str, json_schema: dict[str, Any] | None) -> str:
    """The system text the model actually reads: `system` plus the same
    JSON instruction block `OpenRouterClient`'s manual fallback appends
    (`portolan.llm.openrouter.JSON_INSTRUCTION_TEMPLATE`), or `system`
    unchanged when no schema was requested. `scripts/train/mlx_lora.py`
    builds its training prompts with this same function."""
    if json_schema is None:
        return system
    return system + JSON_INSTRUCTION_TEMPLATE.format(schema=json.dumps(json_schema))


class JsonObjectStop:
    """Watches generated text and reports when the first top-level JSON
    object has closed, so generation can stop there instead of running on
    to `max_tokens`. Braces inside JSON strings are ignored. `json_util.
    extract_json_object` takes the first balanced object anyway, so stopping
    here does not change what gets parsed; it only saves time. A leading
    `<think>...</think>` block is skipped, as `strip_leading_think` skips it
    before parsing."""

    def __init__(self) -> None:
        self.depth = 0
        self.in_string = False
        self.escape = False
        self.started = False
        self.done = False
        self._phase = "start"  # start -> (think ->) answer
        self._lead = ""

    def feed(self, text: str) -> bool:
        if self._phase != "answer":
            self._lead += text
            if self._phase == "start":
                stripped = self._lead.lstrip()
                if not stripped or _THINK_OPEN.startswith(stripped):
                    return False
                self._phase = "think" if stripped.startswith(_THINK_OPEN) else "answer"
                if self._phase == "answer":
                    return self._scan(self._lead)
            end = self._lead.find(_THINK_CLOSE)
            if end < 0:
                return False
            self._phase = "answer"
            return self._scan(self._lead[end + len(_THINK_CLOSE) :])
        return self._scan(text)

    def _scan(self, text: str) -> bool:
        for ch in text:
            if self.done:
                break
            if self.in_string:
                if self.escape:
                    self.escape = False
                elif ch == "\\":
                    self.escape = True
                elif ch == '"':
                    self.in_string = False
                continue
            if ch == '"' and self.depth > 0:
                self.in_string = True
            elif ch == "{":
                self.depth += 1
                self.started = True
            elif ch == "}" and self.depth > 0:
                self.depth -= 1
                if self.started and self.depth == 0:
                    self.done = True
        return self.done


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_leading_think(text: str) -> str:
    """`text` without a leading `<think>...</think>` block. Some local
    models reason before answering even with thinking switched off in the
    chat template; the block is not part of the answer, and braces inside
    it would otherwise be read as the start of the JSON object."""
    return _THINK_BLOCK.sub("", text, count=1)


def read_prefix_mode(adapter_path: str | Path | None) -> str:
    """`adapter_config.json`'s `portolan_prefix_mode`, defaulting to
    `standard` (no adapter, or an adapter trained by stock `mlx_lm lora`)."""
    if adapter_path is None:
        return PREFIX_MODE_STANDARD
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.is_file():
        return PREFIX_MODE_STANDARD
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mode = config.get("portolan_prefix_mode", PREFIX_MODE_STANDARD)
    if mode not in (PREFIX_MODE_STANDARD, PREFIX_MODE_FROZEN_BASE):
        raise ValueError(f"{config_path}: unknown portolan_prefix_mode {mode!r}")
    return mode


@contextmanager
def lora_disabled(model: Any) -> Iterator[int]:
    """Set every LoRA layer's `scale` to 0 for the duration (a `LoRALinear`
    computes `base(x) + scale * lora(x)`, so this is exactly the base
    model), restoring the original scales on exit. Yields how many layers
    were switched off."""
    saved: list[tuple[Any, float]] = []
    for _, module in model.named_modules():
        if hasattr(module, "lora_a") and hasattr(module, "scale"):
            saved.append((module, module.scale))
            module.scale = 0.0
    try:
        yield len(saved)
    finally:
        for module, scale in saved:
            module.scale = scale


class MLXClient:
    """In-process `mlx_lm` model behind the `LLMClient` protocol. One
    instance holds one model in memory; the bench runs it at
    `--concurrency 1`. Metering records real token counts at a cost of 0.0
    (local compute), under model key `mlx:<model id>`."""

    def __init__(
        self,
        model: str,
        *,
        adapter_path: str | Path | None = None,
        revision: str | None = None,
        tokenizer_config: dict[str, Any] | None = None,
        sink: UsageSink | None = None,
        tenant: str = "dev",
        max_tokens: int | None = None,
        progress: bool = False,
    ) -> None:
        self.model_id = model
        self.revision = revision
        self.tokenizer_config = dict(tokenizer_config or {})
        self.progress = progress
        self.calls = 0
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.prefix_mode = read_prefix_mode(self.adapter_path)
        self.sink = sink or InMemorySink()
        self.tenant = tenant
        self.max_tokens_cap = max_tokens
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._prefix_system: str | None = None
        self._prefix_ids: list[int] = []
        self._prefix_cache: list[Any] | None = None

    @property
    def model_key(self) -> str:
        return f"mlx:{self.model_id}"

    # -- loading ---------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not mlx_available():
            raise RuntimeError(
                "MLXClient needs mlx and mlx-lm: run `uv sync --group mlx` on Apple Silicon"
            )
        from mlx_lm import load

        adapter = str(self.adapter_path) if self.adapter_path is not None else None
        self._model, self._tokenizer = load(
            self.model_id,
            adapter_path=adapter,
            revision=self.revision,
            tokenizer_config=self.tokenizer_config or None,
        )

    def unload(self) -> None:
        """Drop the weights and the prefix cache and return MLX's cached
        buffers to the system, so another model can load in the same process
        (a 16 GB machine holds one local model at a time)."""
        if self._model is None:
            return
        self._model = None
        self._tokenizer = None
        self._prefix_system = None
        self._prefix_ids = []
        self._prefix_cache = None
        gc.collect()
        import mlx.core as mx

        mx.clear_cache()

    def load(self) -> None:
        """Load the weights now rather than on the first call."""
        with self._lock:
            self._ensure_loaded()

    def warm(self, system: str, json_schema: dict[str, Any] | None = None) -> None:
        """Load the model and encode the shared prefix for `system` now, so a
        timed run measures per-source work, not the one-time setup."""
        with self._lock:
            self._ensure_loaded()
            self._ensure_prefix(effective_system_prompt(system, json_schema))

    @staticmethod
    def peak_memory_bytes() -> int:
        """MLX's peak memory for this process since the last reset (weights,
        key/value caches and activations); 0 when MLX is absent."""
        if not mlx_available():
            return 0
        import mlx.core as mx

        return int(mx.get_peak_memory())

    def _template_ids(self, messages: list[dict[str, str]], *, generation: bool) -> list[int]:
        ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=generation,
            enable_thinking=False,
            tokenize=True,
            return_dict=False,
        )
        return list(ids)

    def _system_prefix_ids(self, system: str) -> list[int]:
        """The token ids every call with `system` starts with. Normally the
        chat template's system turn alone. A template that cannot render a
        system turn by itself (it folds the system text into the first user
        turn, or refuses a conversation without one) gets the common prefix
        of two full renders instead, minus the last shared token, which a
        tokenizer may merge differently at the boundary. Only the standard
        mode may fall back: a `frozen_base` adapter was trained on the
        system turn exactly."""
        system_turn = [{"role": "system", "content": system}]
        try:
            ids: list[int] | None = self._template_ids(system_turn, generation=False)
        except Exception:  # noqa: BLE001 - a template's own refusal, handled below
            ids = None
        if self.prefix_mode == PREFIX_MODE_FROZEN_BASE:
            if ids is None:
                raise RuntimeError("frozen_base adapter: the chat template has no system turn")
            return ids
        probe = self._template_ids(
            [*system_turn, {"role": "user", "content": "first probe"}], generation=True
        )
        if ids is not None and len(ids) < len(probe) and probe[: len(ids)] == ids:
            return ids
        other = self._template_ids(
            [*system_turn, {"role": "user", "content": "another text"}], generation=True
        )
        shared = 0
        for a, b in zip(probe, other, strict=False):
            if a != b:
                break
            shared += 1
        return probe[: max(0, shared - 1)]

    def _ensure_prefix(self, system: str) -> None:
        """Encode the shared prefix for `system` once into a reusable cache
        (`_prefix_cache` stays `None` when the template shares no prefix)."""
        if self._prefix_system == system:
            return
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        prefix_ids = self._system_prefix_ids(system)
        cache = make_prompt_cache(self._model) if prefix_ids else None
        tokens = mx.array(prefix_ids)[None]

        def _prefill() -> None:
            for start in range(0, len(prefix_ids), _PREFILL_CHUNK):
                self._model(tokens[:, start : start + _PREFILL_CHUNK], cache=cache)
                mx.eval([c.state for c in cache])

        if cache is not None:
            if self.prefix_mode == PREFIX_MODE_FROZEN_BASE:
                with lora_disabled(self._model):
                    _prefill()
            else:
                _prefill()
        self._prefix_system = system
        self._prefix_ids = prefix_ids
        self._prefix_cache = cache

    # -- LLMClient -------------------------------------------------------

    @contextmanager
    def using_sink(self, sink: UsageSink, tenant: str) -> Iterator[None]:
        """The optional `using_sink` extension (see `portolan.llm.base`)."""
        previous = self.sink, self.tenant
        self.sink, self.tenant = sink, tenant
        try:
            yield
        finally:
            self.sink, self.tenant = previous

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        model: str | None = None,
        surface: Surface,
        max_tokens: int = 4096,
        temperature: float | None = None,
        validate_json: bool = True,
    ) -> LLMResult:
        if model is not None and model != self.model_id:
            raise ValueError(
                f"MLXClient holds {self.model_id!r} in memory; got a call for {model!r}"
            )
        with self._lock:
            return self._complete_locked(
                system=system,
                user=user,
                json_schema=json_schema,
                surface=surface,
                max_tokens=max_tokens,
                temperature=temperature,
                validate_json=validate_json,
            )

    def _complete_locked(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None,
        surface: Surface,
        max_tokens: int,
        temperature: float | None,
        validate_json: bool,
    ) -> LLMResult:
        self._ensure_loaded()
        from mlx_lm.generate import stream_generate
        from mlx_lm.models.cache import (
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )
        from mlx_lm.sample_utils import make_sampler

        effective_system = effective_system_prompt(system, json_schema)
        self._ensure_prefix(effective_system)
        full_ids = self._template_ids(
            [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user},
            ],
            generation=True,
        )
        prefix_len = len(self._prefix_ids)
        trim_after = False
        if (
            self._prefix_cache is not None
            and full_ids[:prefix_len] == self._prefix_ids
            and len(full_ids) > prefix_len
        ):
            prompt_ids = full_ids[prefix_len:]
            cached_tokens = prefix_len
            # A key/value cache is trimmed back to the prefix after the call; a
            # recurrent or rotating one (hybrid and sliding-window models)
            # cannot be, so the call runs on a copy and the prefix stays intact.
            trim_after = can_trim_prompt_cache(self._prefix_cache)
            cache = self._prefix_cache if trim_after else copy.deepcopy(self._prefix_cache)
        else:
            if self.prefix_mode == PREFIX_MODE_FROZEN_BASE:
                raise RuntimeError(
                    "frozen_base adapter: the chat template did not reproduce the cached "
                    "system prefix, so the prefix cannot be encoded with the adapter off"
                )
            prompt_ids = full_ids
            cache = make_prompt_cache(self._model)
            cached_tokens = 0

        temp = resolve_temperature(temperature, surface) or 0.0
        limit = max_tokens if self.max_tokens_cap is None else min(max_tokens, self.max_tokens_cap)
        # Only a JSON answer ends at its first closed object; free text (a v2
        # writer draft) runs to the model's own end of turn.
        stop = JsonObjectStop() if json_schema is not None else None
        pieces: list[str] = []
        generated = 0
        finish_reason = "length"
        started = time.perf_counter()
        with metered(self.sink, self.tenant, surface, self.model_key) as usage:
            usage.cost_override = 0.0
            try:
                for response in stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt_ids,
                    max_tokens=limit,
                    sampler=make_sampler(temp=temp),
                    prompt_cache=cache,
                ):
                    pieces.append(response.text)
                    generated = response.generation_tokens
                    if response.finish_reason is not None:
                        finish_reason = response.finish_reason
                    if stop is not None and stop.feed(response.text):
                        finish_reason = "json_complete"
                        break
            finally:
                if trim_after:
                    extra_tokens = cache[0].offset - prefix_len
                    if extra_tokens > 0:
                        trim_prompt_cache(cache, extra_tokens)
            elapsed = time.perf_counter() - started
            usage.input_tokens = len(full_ids)
            usage.cached_tokens = cached_tokens
            usage.output_tokens = generated
            usage.extra["mlx"] = {
                "finish_reason": finish_reason,
                "retry": RETRY_MARKER in user,
                "seconds": round(elapsed, 3),
                "prefix_mode": self.prefix_mode,
                "adapter": str(self.adapter_path) if self.adapter_path else None,
                "peak_memory_bytes": self.peak_memory_bytes(),
            }

        self.calls += 1
        if self.progress:
            # A local run takes minutes per long source; one stderr line per
            # call shows it moving.
            print(
                f"mlx call {self.calls}: in={len(full_ids)} cached={cached_tokens} "
                f"out={generated} {elapsed:.1f}s finish={finish_reason} "
                f"retry={RETRY_MARKER in user}",
                file=sys.stderr,
                flush=True,
            )
        text = strip_leading_think("".join(pieces))
        data = extract_json_object(text)
        if data is not None and json_schema is not None and validate_json:
            if validate_json_object(data, json_schema) is not None:
                data = None
        return LLMResult(
            text=text,
            data=data,
            input_tokens=len(full_ids),
            output_tokens=generated,
            cached_tokens=cached_tokens,
            model=self.model_key,
        )
