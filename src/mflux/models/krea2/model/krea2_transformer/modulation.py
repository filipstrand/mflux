"""AdaLN-single modulation for Krea-2.

``DoubleSharedModulation`` adds a per-block learned bias to the shared
timestep vector and splits it into the six attn/mlp (scale, shift, gate)
chunks. ``SimpleModulation`` produces the (scale, shift) pair for the final
layer.
"""

import mlx.core as mx
from mlx import nn


class DoubleSharedModulation(nn.Module):
    """Per-block ``mod.lin`` bias on the shared timestep vector -> 6 chunks."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = mx.zeros((6 * dim,))

    def __call__(self, vec: mx.array) -> tuple[mx.array, ...]:
        out = vec + self.lin.astype(vec.dtype)
        return mx.split(out, 6, axis=-1)


class SimpleModulation(nn.Module):
    """Final-layer modulation: ``vec + lin`` -> (scale, shift)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = mx.zeros((2, dim))

    def __call__(self, vec: mx.array) -> tuple[mx.array, mx.array]:
        out = vec + self.lin.astype(vec.dtype)[None]
        scale, shift = mx.split(out, 2, axis=1)
        return scale, shift
