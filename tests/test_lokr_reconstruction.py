"""Parity tests for the LoKr Kronecker delta reconstruction (sc-2216 / sc-2314).

Validates the MLX reconstruction against a NumPy ``np.kron`` golden — the same
Kronecker definition PEFT's ``LoKrLayer.get_delta_weight`` uses (``torch.kron``) —
and checks the ``LoKrLinear`` forward residual (scale=0 is an exact no-op).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mflux.models.common.lora.layer.lokr_linear_layer import (
    LoKrLinear,
    reconstruct_lokr_delta,
)


def _golden(alpha, rank, w1, w2, base_shape):
    return (np.kron(w1, w2) * (alpha / rank)).reshape(base_shape).astype(np.float32)


def test_reconstruct_full_w1_full_w2_matches_numpy_kron():
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((2, 3)).astype(np.float32)
    w2 = rng.standard_normal((4, 5)).astype(np.float32)
    alpha, rank = 8.0, 4
    base_shape = (2 * 4, 3 * 5)  # [out, in]

    delta = reconstruct_lokr_delta(
        alpha=alpha, rank=rank, base_shape=base_shape, w1=mx.array(w1), w2=mx.array(w2)
    )
    expected = _golden(alpha, rank, w1, w2, base_shape)
    assert np.allclose(np.array(delta), expected, atol=1e-5)
    assert tuple(delta.shape) == base_shape


def test_reconstruct_factored_w2_matches_numpy_kron():
    # The common PEFT case: w1 full, w2 low-rank (w2_a @ w2_b).
    rng = np.random.default_rng(1)
    w1 = rng.standard_normal((3, 3)).astype(np.float32)
    w2_a = rng.standard_normal((6, 2)).astype(np.float32)
    w2_b = rng.standard_normal((2, 7)).astype(np.float32)
    alpha, rank = 16.0, 8
    base_shape = (3 * 6, 3 * 7)

    delta = reconstruct_lokr_delta(
        alpha=alpha,
        rank=rank,
        base_shape=base_shape,
        w1=mx.array(w1),
        w2_a=mx.array(w2_a),
        w2_b=mx.array(w2_b),
    )
    # Build the golden from the MLX factor product so the assertion isolates the
    # kron+scale+reshape under test, not MLX's matmul precision (Metal runs fp32
    # matmul reduced-precision, ~1e-3 — negligible vs bf16 inference).
    factor2 = np.array(mx.matmul(mx.array(w2_a), mx.array(w2_b)))
    expected = _golden(alpha, rank, w1, factor2, base_shape)
    assert np.allclose(np.array(delta), expected, atol=1e-5)


def test_reconstruct_factored_both_matches_numpy_kron():
    rng = np.random.default_rng(2)
    w1_a = rng.standard_normal((4, 2)).astype(np.float32)
    w1_b = rng.standard_normal((2, 3)).astype(np.float32)
    w2_a = rng.standard_normal((5, 2)).astype(np.float32)
    w2_b = rng.standard_normal((2, 6)).astype(np.float32)
    alpha, rank = 4.0, 4  # scaling 1.0
    base_shape = (4 * 5, 3 * 6)

    delta = reconstruct_lokr_delta(
        alpha=alpha,
        rank=rank,
        base_shape=base_shape,
        w1_a=mx.array(w1_a),
        w1_b=mx.array(w1_b),
        w2_a=mx.array(w2_a),
        w2_b=mx.array(w2_b),
    )
    factor1 = np.array(mx.matmul(mx.array(w1_a), mx.array(w1_b)))
    factor2 = np.array(mx.matmul(mx.array(w2_a), mx.array(w2_b)))
    expected = _golden(alpha, rank, factor1, factor2, base_shape)
    assert np.allclose(np.array(delta), expected, atol=1e-5)


def test_lokr_linear_scale_zero_is_exact_noop():
    rng = np.random.default_rng(3)
    in_dims, out_dims = 12, 8
    base = nn.Linear(in_dims, out_dims, bias=False)
    delta = reconstruct_lokr_delta(
        alpha=8.0,
        rank=4,
        base_shape=(out_dims, in_dims),
        w1=mx.array(rng.standard_normal((2, 3)).astype(np.float32)),
        w2=mx.array(rng.standard_normal((4, 4)).astype(np.float32)),
    )
    layer = LoKrLinear.from_linear(base, delta=delta, scale=0.0)
    x = mx.array(rng.standard_normal((5, in_dims)).astype(np.float32))
    assert np.array_equal(np.array(layer(x)), np.array(base(x)))


def test_lokr_linear_forward_applies_delta_residual():
    rng = np.random.default_rng(4)
    in_dims, out_dims = 12, 8
    base = nn.Linear(in_dims, out_dims, bias=False)
    delta = reconstruct_lokr_delta(
        alpha=8.0,
        rank=4,
        base_shape=(out_dims, in_dims),
        w1=mx.array(rng.standard_normal((2, 3)).astype(np.float32)),
        w2=mx.array(rng.standard_normal((4, 4)).astype(np.float32)),
    )
    scale = 0.7
    layer = LoKrLinear.from_linear(base, delta=delta, scale=scale)
    x = mx.array(rng.standard_normal((5, in_dims)).astype(np.float32))
    # Reference uses the MLX residual matmul too, isolating the __call__ wiring
    # from Metal's reduced-precision fp32 matmul.
    expected = np.array(base(x)) + scale * np.array(mx.matmul(x, delta.T))
    assert np.allclose(np.array(layer(x)), expected, atol=1e-5)
