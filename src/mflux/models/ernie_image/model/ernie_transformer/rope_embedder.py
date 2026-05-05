import mlx.core as mx
from mlx import nn


def _rope(pos: mx.array, dim: int, theta: int) -> mx.array:
    # pos: [...], dim: int, theta: int
    # Returns: [..., dim//2] — raw angle values (not cos/sin yet)
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim  # [dim//2]
    omega = 1.0 / (theta**scale)  # [dim//2]
    # pos[..., None] * omega[None, ...] – broadcast outer product
    return pos[..., None].astype(mx.float32) * omega  # [..., dim//2]


class ErnieRopeEmbedder(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array) -> mx.array:
        # ids: [B, S, 3]
        # Returns: [B, S, 1, head_dim] – raw angles (apply cos/sin in attention)
        parts = [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)]
        emb = mx.concatenate(parts, axis=-1)  # [B, S, sum(axes_dim)//2=64]
        emb = emb[:, :, None, :]  # [B, S, 1, 64]
        # Duplicate each angle to produce [θ₀, θ₀, θ₁, θ₁, ...] pattern → head_dim=128
        emb = mx.stack([emb, emb], axis=-1)  # [B, S, 1, 64, 2]
        return emb.reshape(*emb.shape[:-2], -1)  # [B, S, 1, 128]


def apply_rotary_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
    # q, k: [B, S, heads, head_dim]
    # cos, sin: [B, S, 1, head_dim] — pre-computed once per resolution+prompt
    # rot_dim == head_dim for ERNIE, so q_pass / k_pass are always empty — skip the split.
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rotated = mx.concatenate([-q2, q1], axis=-1)
    k_rotated = mx.concatenate([-k2, k1], axis=-1)
    return q * cos + q_rotated * sin, k * cos + k_rotated * sin
