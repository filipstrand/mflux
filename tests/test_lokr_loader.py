"""Mechanical loader test for MLX-native LoKr (sc-2216 / sc-2314).

Synthesizes a LoKr adapter in the on-disk key layout SceneWorks' write_lokr_adapter
produces (``‹module›.lokr_w1``/``lokr_w2`` + ``networkType``/``alpha``/``rank`` metadata),
then drives it through LoKrLoader onto a tiny mflux-style module tree: confirms the
targeted Linear is replaced with a LoKrLinear, the metadata round-trips through
``mx.load`` (so LoRALoader's routing fires), and the scale knob behaves (0 = no-op).
"""

import os
import tempfile

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.layer.lokr_linear_layer import LoKrLinear, reconstruct_lokr_delta
from mflux.models.common.lora.mapping.lokr_loader import LoKrLoader


class _Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = _Attention(dim)


class _Tiny(nn.Module):
    def __init__(self, dim, n=2):
        super().__init__()
        self.layers = [_Block(dim) for _ in range(n)]


def _lokr_weights(dim, rng):
    # dim = a*a with full factors a×a so kron(w1,w2) = [dim, dim].
    a = int(round(dim ** 0.5))
    assert a * a == dim
    return {
        "layers.0.attention.to_q.lokr_w1": mx.array(rng.standard_normal((a, a)).astype(np.float32)),
        "layers.0.attention.to_q.lokr_w2": mx.array(rng.standard_normal((a, a)).astype(np.float32)),
    }


def test_is_lokr_detects_metadata():
    assert LoKrLoader.is_lokr({"networkType": "lokr"})
    assert LoKrLoader.is_lokr({"networkType": "LoKr"})
    assert not LoKrLoader.is_lokr({"networkType": "lora"})
    assert not LoKrLoader.is_lokr({})
    assert not LoKrLoader.is_lokr(None)


def test_apply_replaces_target_with_lokr_linear():
    rng = np.random.default_rng(0)
    dim = 16
    model = _Tiny(dim)
    weights = _lokr_weights(dim, rng)
    metadata = {"networkType": "lokr", "alpha": "8", "rank": "4"}

    applied, matched = LoKrLoader.apply(model, weights, metadata, scale=1.0)

    assert applied == 1
    assert matched == set(weights.keys())
    assert isinstance(model.layers[0].attention.to_q, LoKrLinear)
    # Untargeted module is untouched.
    assert isinstance(model.layers[1].attention.to_q, nn.Linear)


def test_scale_zero_is_noop_and_scale_one_changes_output():
    rng = np.random.default_rng(1)
    dim = 16
    weights = _lokr_weights(dim, rng)
    metadata = {"networkType": "lokr", "alpha": "8", "rank": "4"}
    x = mx.array(rng.standard_normal((3, dim)).astype(np.float32))

    base_model = _Tiny(dim)
    base_out = np.array(base_model.layers[0].attention.to_q(x))

    noop = _Tiny(dim)
    noop.layers[0].attention.to_q = base_model.layers[0].attention.to_q  # share base weights
    LoKrLoader.apply(noop, weights, metadata, scale=0.0)
    assert np.array_equal(np.array(noop.layers[0].attention.to_q(x)), base_out)

    active = _Tiny(dim)
    active.layers[0].attention.to_q = base_model.layers[0].attention.to_q  # same base weights
    LoKrLoader.apply(active, weights, metadata, scale=1.0)
    out = np.array(active.layers[0].attention.to_q(x))
    assert not np.allclose(out, base_out)


def test_delta_stored_at_bf16():
    # The reconstructed ΔW is downcast to bf16 in the layer (memory win).
    rng = np.random.default_rng(5)
    dim = 16
    model = _Tiny(dim)
    LoKrLoader.apply(model, _lokr_weights(dim, rng), {"networkType": "lokr", "alpha": "8", "rank": "4"}, scale=1.0)
    assert model.layers[0].attention.to_q.delta.dtype == mx.bfloat16


def test_stacking_two_lokr_fuses_and_sums():
    rng = np.random.default_rng(6)
    dim = 16
    metadata = {"networkType": "lokr", "alpha": "8", "rank": "4"}
    model = _Tiny(dim)
    base = model.layers[0].attention.to_q  # original nn.Linear, captured before mutation
    x = mx.array(rng.standard_normal((3, dim)).astype(np.float32))
    base_out = np.array(base(x))

    LoKrLoader.apply(model, _lokr_weights(dim, rng), metadata, scale=1.0)
    LoKrLoader.apply(model, _lokr_weights(dim, rng), metadata, scale=1.0)

    fused = model.layers[0].attention.to_q
    assert isinstance(fused, FusedLoRALinear)
    assert len(fused.loras) == 2
    assert fused.base_linear is base  # stacked, not re-wrapped
    # Output = shared base applied ONCE + both LoKr residuals.
    expected = base_out + np.array(fused.loras[0].residual(x)) + np.array(fused.loras[1].residual(x))
    assert np.allclose(np.array(fused(x)), expected, atol=1e-2)


def test_lokr_stacks_onto_existing_lora():
    # Pre-install a LoRA, then apply a LoKr to the same module → mixed fusion.
    rng = np.random.default_rng(7)
    dim = 16
    model = _Tiny(dim)
    base = model.layers[0].attention.to_q
    model.layers[0].attention.to_q = LoRALinear.from_linear(base, r=4, scale=1.0)

    applied, _ = LoKrLoader.apply(model, _lokr_weights(dim, rng), {"networkType": "lokr", "alpha": "8", "rank": "4"}, scale=1.0)

    fused = model.layers[0].attention.to_q
    assert applied == 1
    assert isinstance(fused, FusedLoRALinear)
    assert isinstance(fused.loras[0], LoRALinear)
    assert isinstance(fused.loras[1], LoKrLinear)
    assert fused.base_linear is base


def test_fused_mixes_lora_and_lokr_residuals():
    # Layer-level: a FusedLoRALinear with one LoRA and one LoKr sums both residuals
    # on top of a single base application.
    rng = np.random.default_rng(8)
    dim = 16
    base = nn.Linear(dim, dim, bias=False)
    lora = LoRALinear.from_linear(base, r=4, scale=0.5)
    lokr = LoKrLinear.from_linear(
        base,
        delta=reconstruct_lokr_delta(
            alpha=8.0, rank=4, base_shape=(dim, dim),
            w1=mx.array(rng.standard_normal((4, 4)).astype(np.float32)),
            w2=mx.array(rng.standard_normal((4, 4)).astype(np.float32)),
        ),
        scale=0.7,
    )
    fused = FusedLoRALinear(base_linear=base, loras=[lora, lokr])
    x = mx.array(rng.standard_normal((3, dim)).astype(np.float32))
    expected = np.array(base(x)) + np.array(lora.residual(x)) + np.array(lokr.residual(x))
    assert np.allclose(np.array(fused(x)), expected, atol=1e-2)


def test_metadata_round_trips_through_mx_load():
    rng = np.random.default_rng(2)
    dim = 16
    weights = _lokr_weights(dim, rng)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "char_lokr.safetensors")
        mx.save_safetensors(path, weights, metadata={"networkType": "lokr", "alpha": "8", "rank": "4"})
        loaded = mx.load(path, return_metadata=True)
        arrays, meta = dict(loaded[0].items()), loaded[1]
        assert LoKrLoader.is_lokr(meta)
        model = _Tiny(dim)
        applied, _ = LoKrLoader.apply(model, arrays, meta, scale=1.0)
        assert applied == 1
        assert isinstance(model.layers[0].attention.to_q, LoKrLinear)
