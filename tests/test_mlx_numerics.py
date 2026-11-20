"""MLX numerics for the fine-tune shakedown (2026-09-22): the frozen-prefix
cache `scripts/train/mlx_lora.py` trains with gives the same logits as
running the whole sequence, and `lora_disabled` really is the base model.
Uses a tiny randomly initialised Qwen3 built in memory -- no download.

Skipped at collection when `mlx`/`mlx_lm` are not installed (CI and the
Docker image never install the optional `mlx` group) or when `scripts/`
is absent (the container image excludes it)."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from tests._paths import require_repo_path

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

_SCRIPT = require_repo_path("scripts", "train", "mlx_lora.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("mlx_lora_script_numerics", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mlx_lora_script_numerics"] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _tiny_model():
    from mlx_lm.models.qwen3 import Model, ModelArgs

    mx.random.seed(0)
    args = ModelArgs(
        model_type="qwen3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=97,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
        head_dim=16,
        tie_word_embeddings=True,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def test_frozen_prefix_cache_matches_the_full_sequence() -> None:
    model = _tiny_model()
    ids = mx.random.randint(0, 97, (1, 40))
    prefix = ids[0, :29].tolist()
    full = model(ids)[:, 29:]
    kv = script.encode_prefix(model, prefix)
    cache = [script.FrozenPrefixKV(k, v) for k, v in kv]
    part = model(ids[:, 29:], cache=cache)
    assert mx.allclose(full, part, atol=1e-4).item()


def test_lora_disabled_is_exactly_the_base_model() -> None:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from portolan.llm.mlx_local import lora_disabled

    model = _tiny_model()
    ids = mx.random.randint(0, 97, (1, 12))
    base = model(ids)
    model.freeze()
    linear_to_lora_layers(model, 2, {"rank": 4, "dropout": 0.0, "scale": 20.0})
    for _, module in model.named_modules():
        if hasattr(module, "lora_b"):
            module.lora_b = mx.random.normal(module.lora_b.shape) * 0.1
    tuned = model(ids)
    assert not mx.allclose(base, tuned, atol=1e-3).item()
    with lora_disabled(model) as switched:
        assert switched > 0
        assert mx.allclose(base, model(ids), atol=1e-5).item()
    assert mx.allclose(tuned, model(ids), atol=1e-5).item()


def test_frozen_prefix_loss_trains_the_adapter() -> None:
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model = _tiny_model()
    kv = script.encode_prefix(model, list(range(20)))
    model.freeze()
    linear_to_lora_layers(model, 2, {"rank": 4, "dropout": 0.0, "scale": 20.0})

    def loss(model, batch):
        cache = [script.FrozenPrefixKV(k, v) for k, v in kv]
        logits = model(batch[:, :-1], cache=cache)
        return nn.losses.cross_entropy(logits, batch[:, 1:]).mean()

    _, grads = nn.value_and_grad(model, loss)(model, mx.random.randint(0, 97, (1, 9)))
    flat = dict(tree_flatten(grads))
    assert flat and all(k.endswith(("lora_a", "lora_b")) for k in flat)
    assert any(mx.abs(v).sum().item() > 0 for v in flat.values())
