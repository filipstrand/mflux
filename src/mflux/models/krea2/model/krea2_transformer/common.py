"""Shared primitives for the Krea-2 single-stream DiT.

Krea-2 normalisation follows the reference ``(1 + scale)`` convention: the
``scale`` parameter is stored zero-centred and ``1.0`` is added at apply time,
with the reduction performed in float32 for stability.
"""

import mlx.core as mx
from mlx import nn


class Krea2RMSNorm(nn.Module):
    """RMSNorm with the reference ``(1 + scale)`` weight convention (fp32 reduction)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        weight = (self.scale.astype(mx.float32) + 1.0).astype(mx.float32)
        return mx.fast.rms_norm(x.astype(mx.float32), weight, self.eps).astype(dtype)


class Krea2QKNorm(nn.Module):
    """Per-head-dim query/key RMSNorm pair."""

    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.qnorm = Krea2RMSNorm(head_dim, eps=eps)
        self.knorm = Krea2RMSNorm(head_dim, eps=eps)

    def __call__(self, q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
        return self.qnorm(q), self.knorm(k)
