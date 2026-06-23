"""3-axis RoPE for Krea-2.

Identical math to Flux's ``EmbedND`` / ``apply_rope`` (adjacent-pair rotation
with a ``[cos, -sin, sin, cos]`` 2x2 block), but with Krea-2's axes and theta:
``head_dim = 128``, ``axes_dim = [32, 48, 48]`` (temporal, height, width),
``theta = 1000``. Position ids are ``(B, N, 3)``: text tokens at ``(0, 0, 0)``,
image tokens at ``(0, h_idx, w_idx)``.
"""

import mlx.core as mx
from mlx import nn


class Krea2RopeEmbedder(nn.Module):
    def __init__(self, head_dim: int = 128, theta: int = 1000, axes_dim: list[int] | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.axes_dim = axes_dim or [32, 48, 48]
        assert sum(self.axes_dim) == head_dim, f"axes {self.axes_dim} sum != head_dim {head_dim}"

    def __call__(self, ids: mx.array) -> mx.array:
        # ids: (B, N, 3) -> freqs_cis: (B, 1, N, head_dim // 2, 2, 2)
        emb = mx.concatenate(
            [Krea2RopeEmbedder._rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)],
            axis=-3,
        )
        return mx.expand_dims(emb, axis=1)

    @staticmethod
    def _rope(pos: mx.array, dim: int, theta: float) -> mx.array:
        scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        omega = 1.0 / (theta**scale)
        out = mx.expand_dims(pos, axis=-1).astype(mx.float32) * mx.expand_dims(omega, axis=0)
        cos, sin = mx.cos(out), mx.sin(out)
        stacked = mx.stack([cos, -sin, sin, cos], axis=-1)
        return mx.reshape(stacked, (*pos.shape, dim // 2, 2, 2))


def apply_rope(xq: mx.array, xk: mx.array, freqs_cis: mx.array) -> tuple[mx.array, mx.array]:
    xq_ = xq.astype(mx.float32).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.astype(mx.float32).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape), xk_out.reshape(*xk.shape)
