import mlx.core as mx
import numpy as np
from mlx import nn


class Flux3PosEmbed(nn.Module):
    def __init__(self, theta: int = 10000, axes_dim: tuple[int, ...] = (32, 32, 32, 32)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        # Angles in float64 on the host (the GPU has none), as the reference does; cos/sin are float32.
        pos = np.array(ids.astype(mx.float32), dtype=np.float64)
        cos_out, sin_out = [], []
        for i, dim in enumerate(self.axes_dim):
            omega = 1.0 / (float(self.theta) ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
            angles = pos[..., i : i + 1] * omega
            cos_out.append(np.cos(angles))
            sin_out.append(np.sin(angles))
        cos = np.concatenate(cos_out, axis=-1).astype(np.float32)
        sin = np.concatenate(sin_out, axis=-1).astype(np.float32)
        return mx.array(cos), mx.array(sin)

    @staticmethod
    def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x: (B, H, L, D) in float32; cos/sin: (L, D/2); adjacent pairs (x0, x1) rotate together.
        batch, heads, length, dim = x.shape
        pairs = x.reshape(batch, heads, length, dim // 2, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        out = mx.stack([cos * x0 - sin * x1, sin * x0 + cos * x1], axis=-1)
        return out.reshape(batch, heads, length, dim)
