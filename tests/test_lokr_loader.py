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

from mflux.models.common.lora.layer.lokr_linear_layer import LoKrLinear
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
